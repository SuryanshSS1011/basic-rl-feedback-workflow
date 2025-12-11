"""
Security Weights for CVSS/CWE Severity Mapping

Maps security vulnerabilities to normalized severity weights for reward computation.
Weights are derived from CVSS scores and CWE classifications.

Severity normalization:
- Critical (CVSS 9.0-10.0): weight 0.9-1.0
- High (CVSS 7.0-8.9): weight 0.7-0.9
- Medium (CVSS 4.0-6.9): weight 0.4-0.7
- Low (CVSS 0.1-3.9): weight 0.1-0.4
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from pathlib import Path


@dataclass
class VulnerabilityInfo:
    """Information about a vulnerability type"""
    cwe_id: str
    name: str
    weight: float  # Normalized severity weight [0, 1]
    cvss_base: float  # Base CVSS score
    category: str  # critical, high, medium, low


class SecurityWeights:
    """
    CVSS/CWE database for mapping security findings to severity weights.

    Usage:
        weights = SecurityWeights()
        severity = weights.get_weight('CWE-119')  # Returns 0.95
        severity = weights.get_weight_for_codeql('cpp/overflow-buffer')  # Returns 0.95
    """

    # Default CWE severity weights (aligned with CVSS scores)
    DEFAULT_CWE_WEIGHTS: Dict[str, VulnerabilityInfo] = {
        # Critical (CVSS 9.0-10.0)
        'CWE-78': VulnerabilityInfo('CWE-78', 'OS Command Injection', 1.0, 9.8, 'critical'),
        'CWE-89': VulnerabilityInfo('CWE-89', 'SQL Injection', 1.0, 9.8, 'critical'),
        'CWE-94': VulnerabilityInfo('CWE-94', 'Code Injection', 1.0, 9.8, 'critical'),
        'CWE-119': VulnerabilityInfo('CWE-119', 'Buffer Overflow', 0.95, 9.3, 'critical'),
        'CWE-787': VulnerabilityInfo('CWE-787', 'Out-of-bounds Write', 0.95, 9.3, 'critical'),

        # High (CVSS 7.0-8.9)
        'CWE-120': VulnerabilityInfo('CWE-120', 'Buffer Copy without Size Check', 0.85, 8.1, 'high'),
        'CWE-122': VulnerabilityInfo('CWE-122', 'Heap Overflow', 0.85, 8.1, 'high'),
        'CWE-125': VulnerabilityInfo('CWE-125', 'Out-of-bounds Read', 0.80, 7.5, 'high'),
        'CWE-190': VulnerabilityInfo('CWE-190', 'Integer Overflow', 0.75, 7.3, 'high'),
        'CWE-416': VulnerabilityInfo('CWE-416', 'Use After Free', 0.90, 8.8, 'high'),
        'CWE-476': VulnerabilityInfo('CWE-476', 'NULL Pointer Dereference', 0.70, 7.0, 'high'),
        'CWE-415': VulnerabilityInfo('CWE-415', 'Double Free', 0.85, 8.1, 'high'),
        'CWE-362': VulnerabilityInfo('CWE-362', 'Race Condition', 0.75, 7.5, 'high'),
        'CWE-22': VulnerabilityInfo('CWE-22', 'Path Traversal', 0.80, 7.8, 'high'),
        'CWE-434': VulnerabilityInfo('CWE-434', 'Unrestricted Upload', 0.85, 8.0, 'high'),

        # Medium (CVSS 4.0-6.9)
        'CWE-457': VulnerabilityInfo('CWE-457', 'Uninitialized Variable', 0.50, 5.5, 'medium'),
        'CWE-401': VulnerabilityInfo('CWE-401', 'Memory Leak', 0.45, 5.0, 'medium'),
        'CWE-562': VulnerabilityInfo('CWE-562', 'Return of Stack Variable Address', 0.55, 5.8, 'medium'),
        'CWE-134': VulnerabilityInfo('CWE-134', 'Format String Vulnerability', 0.65, 6.5, 'medium'),
        'CWE-369': VulnerabilityInfo('CWE-369', 'Divide By Zero', 0.40, 4.5, 'medium'),
        'CWE-252': VulnerabilityInfo('CWE-252', 'Unchecked Return Value', 0.45, 5.0, 'medium'),
        'CWE-772': VulnerabilityInfo('CWE-772', 'Resource Leak', 0.45, 5.0, 'medium'),
        'CWE-200': VulnerabilityInfo('CWE-200', 'Information Exposure', 0.50, 5.5, 'medium'),

        # Low (CVSS 0.1-3.9)
        'CWE-561': VulnerabilityInfo('CWE-561', 'Dead Code', 0.20, 2.0, 'low'),
        'CWE-563': VulnerabilityInfo('CWE-563', 'Unused Variable', 0.15, 1.5, 'low'),
        'CWE-570': VulnerabilityInfo('CWE-570', 'Expression Always False', 0.15, 1.5, 'low'),
        'CWE-571': VulnerabilityInfo('CWE-571', 'Expression Always True', 0.15, 1.5, 'low'),
        'CWE-478': VulnerabilityInfo('CWE-478', 'Missing Default Case', 0.20, 2.0, 'low'),
        'CWE-480': VulnerabilityInfo('CWE-480', 'Use of Incorrect Operator', 0.25, 2.5, 'low'),
    }

    # CodeQL rule ID to CWE mapping
    CODEQL_TO_CWE: Dict[str, str] = {
        # Buffer/Memory errors
        'cpp/overflow-buffer': 'CWE-119',
        'cpp/unbounded-write': 'CWE-787',
        'cpp/buffer-overflow': 'CWE-120',
        'cpp/stack-buffer-overflow': 'CWE-121',
        'cpp/heap-buffer-overflow': 'CWE-122',
        'cpp/out-of-bounds-read': 'CWE-125',

        # Pointer errors
        'cpp/null-dereference': 'CWE-476',
        'cpp/use-after-free': 'CWE-416',
        'cpp/double-free': 'CWE-415',
        'cpp/return-stack-address': 'CWE-562',

        # Integer errors
        'cpp/integer-overflow': 'CWE-190',
        'cpp/signed-unsigned-comparison': 'CWE-195',

        # Memory management
        'cpp/memory-leak': 'CWE-401',
        'cpp/resource-leak': 'CWE-772',

        # Injection
        'cpp/command-injection': 'CWE-78',
        'cpp/sql-injection': 'CWE-89',
        'cpp/code-injection': 'CWE-94',
        'cpp/format-string': 'CWE-134',

        # Other
        'cpp/uninitialized-variable': 'CWE-457',
        'cpp/unchecked-return': 'CWE-252',
        'cpp/divide-by-zero': 'CWE-369',
        'cpp/dead-code': 'CWE-561',
        'cpp/unused-variable': 'CWE-563',
        'cpp/path-traversal': 'CWE-22',
        'cpp/race-condition': 'CWE-362',

        # Python (Bandit) rules - comprehensive mapping
        'py/command-injection': 'CWE-78',
        'py/sql-injection': 'CWE-89',
        'py/code-injection': 'CWE-94',
        'py/path-traversal': 'CWE-22',

        # Bandit test IDs (B1xx - B7xx)
        # B1xx - Misc tests
        'B101': 'CWE-703',  # assert_used
        'B102': 'CWE-78',   # exec_used
        'B103': 'CWE-732',  # set_bad_file_permissions
        'B104': 'CWE-134',  # hardcoded_bind_all_interfaces
        'B105': 'CWE-259',  # hardcoded_password_string
        'B106': 'CWE-259',  # hardcoded_password_funcarg
        'B107': 'CWE-259',  # hardcoded_password_default
        'B108': 'CWE-377',  # hardcoded_tmp_directory
        'B110': 'CWE-89',   # try_except_pass
        'B112': 'CWE-703',  # try_except_continue

        # B2xx - Shell injection
        'B201': 'CWE-94',   # flask_debug_true
        'B202': 'CWE-502',  # tarfile_unsafe_members

        # B3xx - Blacklist calls
        'B301': 'CWE-502',  # pickle
        'B302': 'CWE-502',  # marshal
        'B303': 'CWE-327',  # md5/sha1 (weak hash)
        'B304': 'CWE-327',  # ciphers (weak crypto)
        'B305': 'CWE-327',  # cipher_modes
        'B306': 'CWE-377',  # mktemp_q
        'B307': 'CWE-94',   # eval
        'B308': 'CWE-79',   # mark_safe (XSS)
        'B309': 'CWE-327',  # httpsconnection
        'B310': 'CWE-22',   # urllib_urlopen
        'B311': 'CWE-330',  # random (not crypto-secure)
        'B312': 'CWE-295',  # telnetlib
        'B313': 'CWE-611',  # xml_bad_cElementTree
        'B314': 'CWE-611',  # xml_bad_ElementTree
        'B315': 'CWE-611',  # xml_bad_expatreader
        'B316': 'CWE-611',  # xml_bad_expatbuilder
        'B317': 'CWE-611',  # xml_bad_sax
        'B318': 'CWE-611',  # xml_bad_minidom
        'B319': 'CWE-611',  # xml_bad_pulldom
        'B320': 'CWE-611',  # xml_bad_etree
        'B321': 'CWE-330',  # ftplib
        'B322': 'CWE-78',   # input (Python 2)
        'B323': 'CWE-295',  # unverified_context
        'B324': 'CWE-327',  # hashlib_new_insecure

        # B4xx - Blacklist imports
        'B401': 'CWE-295',  # import_telnetlib
        'B402': 'CWE-330',  # import_ftplib
        'B403': 'CWE-502',  # import_pickle
        'B404': 'CWE-78',   # import_subprocess
        'B405': 'CWE-611',  # import_xml_etree
        'B406': 'CWE-611',  # import_xml_sax
        'B407': 'CWE-611',  # import_xml_expat
        'B408': 'CWE-611',  # import_xml_minidom
        'B409': 'CWE-611',  # import_xml_pulldom
        'B410': 'CWE-611',  # import_lxml
        'B411': 'CWE-611',  # import_xmlrpclib
        'B412': 'CWE-295',  # import_httpoxy
        'B413': 'CWE-327',  # import_pycrypto

        # B5xx - Cryptography
        'B501': 'CWE-295',  # request_with_no_cert_validation
        'B502': 'CWE-295',  # ssl_with_bad_version
        'B503': 'CWE-295',  # ssl_with_bad_defaults
        'B504': 'CWE-295',  # ssl_with_no_version
        'B505': 'CWE-327',  # weak_cryptographic_key
        'B506': 'CWE-1321', # yaml_load
        'B507': 'CWE-295',  # ssh_no_host_key_verification

        # B6xx - Injection
        'B601': 'CWE-78',   # paramiko_calls
        'B602': 'CWE-78',   # subprocess_popen_with_shell_equals_true
        'B603': 'CWE-78',   # subprocess_without_shell_equals_true
        'B604': 'CWE-78',   # any_other_function_with_shell_equals_true
        'B605': 'CWE-78',   # start_process_with_a_shell
        'B606': 'CWE-78',   # start_process_with_no_shell
        'B607': 'CWE-78',   # start_process_with_partial_path
        'B608': 'CWE-89',   # hardcoded_sql_expressions
        'B609': 'CWE-78',   # linux_commands_wildcard_injection
        'B610': 'CWE-94',   # django_extra_used
        'B611': 'CWE-94',   # django_rawsql_used

        # B7xx - XSS/SSTI
        'B701': 'CWE-79',   # jinja2_autoescape_false
        'B702': 'CWE-79',   # use_of_mako_templates
        'B703': 'CWE-94',   # django_mark_safe
    }

    # Bandit severity weights (direct mapping)
    BANDIT_SEVERITY_WEIGHTS: Dict[str, float] = {
        'high': 0.85,
        'medium': 0.50,
        'low': 0.25,
    }

    # KLEE bug type to weight mapping
    KLEE_BUG_WEIGHTS: Dict[str, float] = {
        'ptr': 0.80,           # Pointer errors
        'free': 0.85,          # Double free / invalid free
        'memory': 0.75,        # Memory errors
        'overflow': 0.85,      # Buffer overflow
        'div': 0.40,           # Division by zero
        'assert': 0.30,        # Assertion failure
        'abort': 0.50,         # Abort called
        'exec': 0.60,          # Execution error
        'model': 0.20,         # Model error (less severe)
    }

    def __init__(
        self,
        custom_weights_path: Optional[Path] = None,
        klee_bug_weight: float = 0.8
    ):
        """
        Initialize security weights.

        Args:
            custom_weights_path: Path to custom JSON weights file
            klee_bug_weight: Default weight for KLEE bugs not in mapping
        """
        self.cwe_weights = dict(self.DEFAULT_CWE_WEIGHTS)
        self.codeql_to_cwe = dict(self.CODEQL_TO_CWE)
        self.klee_weights = dict(self.KLEE_BUG_WEIGHTS)
        self.default_klee_weight = klee_bug_weight
        self.default_weight = 0.5  # Default for unknown vulnerabilities

        if custom_weights_path:
            self._load_custom_weights(custom_weights_path)

    def _load_custom_weights(self, path: Path):
        """Load custom weights from JSON file"""
        with open(path) as f:
            data = json.load(f)

        if 'cwe_weights' in data:
            for cwe_id, info in data['cwe_weights'].items():
                self.cwe_weights[cwe_id] = VulnerabilityInfo(
                    cwe_id=cwe_id,
                    name=info.get('name', 'Unknown'),
                    weight=info['weight'],
                    cvss_base=info.get('cvss_base', 5.0),
                    category=info.get('category', 'medium'),
                )

        if 'codeql_to_cwe' in data:
            self.codeql_to_cwe.update(data['codeql_to_cwe'])

        if 'klee_weights' in data:
            self.klee_weights.update(data['klee_weights'])

    def get_weight(self, identifier: str) -> float:
        """
        Get severity weight for a CWE ID.

        Args:
            identifier: CWE ID (e.g., 'CWE-119')

        Returns:
            Normalized weight in [0, 1]
        """
        # Normalize identifier
        if not identifier.startswith('CWE-'):
            identifier = f'CWE-{identifier}'

        if identifier in self.cwe_weights:
            return self.cwe_weights[identifier].weight

        return self.default_weight

    def get_weight_for_codeql(self, rule_id: str) -> float:
        """
        Get severity weight for a CodeQL rule ID.

        Args:
            rule_id: CodeQL rule ID (e.g., 'cpp/overflow-buffer')

        Returns:
            Normalized weight in [0, 1]
        """
        # Map CodeQL rule to CWE
        cwe_id = self.codeql_to_cwe.get(rule_id)
        if cwe_id:
            return self.get_weight(cwe_id)

        # Check if rule_id contains severity hints
        if 'critical' in rule_id.lower() or 'injection' in rule_id.lower():
            return 0.9
        elif 'overflow' in rule_id.lower() or 'buffer' in rule_id.lower():
            return 0.85
        elif 'null' in rule_id.lower() or 'pointer' in rule_id.lower():
            return 0.7
        elif 'leak' in rule_id.lower():
            return 0.45
        elif 'unused' in rule_id.lower() or 'dead' in rule_id.lower():
            return 0.2

        return self.default_weight

    def get_weight_for_bandit(
        self,
        test_id: str,
        severity: str = "medium"
    ) -> float:
        """
        Get severity weight for a Bandit finding.

        Args:
            test_id: Bandit test ID (e.g., 'B102', 'B608')
            severity: Bandit severity level (low, medium, high)

        Returns:
            Normalized weight in [0, 1]
        """
        # First try to map to CWE
        cwe_id = self.codeql_to_cwe.get(test_id)
        if cwe_id:
            return self.get_weight(cwe_id)

        # Fall back to severity-based weight
        severity_lower = severity.lower()
        if severity_lower in self.BANDIT_SEVERITY_WEIGHTS:
            return self.BANDIT_SEVERITY_WEIGHTS[severity_lower]

        return self.default_weight

    def get_weight_for_klee(self, bug_type: str) -> float:
        """
        Get severity weight for a KLEE bug type.

        Args:
            bug_type: KLEE bug type (e.g., 'ptr', 'free', 'overflow')

        Returns:
            Normalized weight in [0, 1]
        """
        bug_type_lower = bug_type.lower()

        # Direct lookup
        if bug_type_lower in self.klee_weights:
            return self.klee_weights[bug_type_lower]

        # Partial match
        for key, weight in self.klee_weights.items():
            if key in bug_type_lower:
                return weight

        return self.default_klee_weight

    def compute_normalized_severity(
        self,
        codeql_findings: List[Dict],
        klee_bugs: Optional[List[Dict]] = None,
        max_severity: float = 5.0
    ) -> float:
        """
        Compute normalized severity score V from all findings.

        V = Σ(wi · si) / max_severity

        Args:
            codeql_findings: List of CodeQL findings with 'rule_id' key
            klee_bugs: List of KLEE bugs with 'type' key
            max_severity: Maximum possible severity for normalization

        Returns:
            Normalized severity V in [0, 1]
        """
        total_weighted_severity = 0.0

        # CodeQL findings
        for finding in codeql_findings:
            rule_id = finding.get('rule_id', finding.get('ruleId', ''))
            weight = self.get_weight_for_codeql(rule_id)
            total_weighted_severity += weight

        # KLEE bugs
        if klee_bugs:
            for bug in klee_bugs:
                bug_type = bug.get('type', bug.get('bug_type', 'unknown'))
                weight = self.get_weight_for_klee(bug_type)
                total_weighted_severity += weight

        # Normalize
        normalized_v = total_weighted_severity / max_severity
        return min(normalized_v, 1.0)  # Cap at 1.0

    def get_vulnerability_info(self, cwe_id: str) -> Optional[VulnerabilityInfo]:
        """Get full vulnerability info for a CWE ID"""
        if not cwe_id.startswith('CWE-'):
            cwe_id = f'CWE-{cwe_id}'
        return self.cwe_weights.get(cwe_id)

    def list_all_cwes(self) -> List[str]:
        """List all known CWE IDs"""
        return list(self.cwe_weights.keys())

    def list_by_category(self, category: str) -> List[VulnerabilityInfo]:
        """List vulnerabilities by category (critical, high, medium, low)"""
        return [
            info for info in self.cwe_weights.values()
            if info.category == category.lower()
        ]

    def to_json(self) -> str:
        """Export weights to JSON string"""
        data = {
            'cwe_weights': {
                cwe_id: {
                    'name': info.name,
                    'weight': info.weight,
                    'cvss_base': info.cvss_base,
                    'category': info.category,
                }
                for cwe_id, info in self.cwe_weights.items()
            },
            'codeql_to_cwe': self.codeql_to_cwe,
            'klee_weights': self.klee_weights,
        }
        return json.dumps(data, indent=2)

    def save(self, path: Path):
        """Save weights to JSON file"""
        with open(path, 'w') as f:
            f.write(self.to_json())
