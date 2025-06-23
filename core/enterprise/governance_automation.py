"""
Enterprise Governance Policy Automation Module

This module provides comprehensive governance automation capabilities including
policy definition, compliance monitoring, audit trails, and automated enforcement.
"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import threading
import schedule
import time
from pathlib import Path

class PolicyType(Enum):
    DATA_PRIVACY = "data_privacy"
    MODEL_BIAS = "model_bias"
    DATA_QUALITY = "data_quality"
    SECURITY = "security"
    RETENTION = "retention"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"

class PolicySeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PolicyStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    ARCHIVED = "archived"

class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    UNKNOWN = "unknown"

@dataclass
class PolicyRule:
    id: str
    name: str
    description: str
    policy_type: PolicyType
    severity: PolicySeverity
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    status: PolicyStatus
    created_at: datetime
    updated_at: datetime
    created_by: str
    tags: List[str]

@dataclass
class ComplianceEvent:
    id: str
    policy_id: str
    resource_id: str
    resource_type: str
    status: ComplianceStatus
    details: Dict[str, Any]
    timestamp: datetime
    remediation_actions: List[str]

@dataclass
class AuditEntry:
    id: str
    event_type: str
    resource_id: str
    resource_type: str
    user_id: str
    action: str
    details: Dict[str, Any]
    timestamp: datetime
    ip_address: Optional[str]
    session_id: Optional[str]

class GovernanceAutomationEngine:
    """
    Comprehensive governance automation engine for enterprise ML platforms.
    """
    
    def __init__(self, db_path: str = "governance.db", mlops_integration: Any = None):
        self.db_path = db_path
        self.mlops_integration = mlops_integration
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._scheduler_thread = None
        self._running = False
        
        self._init_database()
        self._load_default_policies()
    
    def _init_database(self):
        """Initialize the governance database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Policy rules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS policy_rules (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                policy_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                conditions TEXT NOT NULL,
                actions TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                created_by TEXT NOT NULL,
                tags TEXT
            )
        ''')
        
        # Compliance events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_events (
                id TEXT PRIMARY KEY,
                policy_id TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                status TEXT NOT NULL,
                details TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                remediation_actions TEXT,
                FOREIGN KEY (policy_id) REFERENCES policy_rules (id)
            )
        ''')
        
        # Audit trail table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_trail (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                ip_address TEXT,
                session_id TEXT
            )
        ''')
        
        # Compliance summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_summary (
                resource_id TEXT PRIMARY KEY,
                resource_type TEXT NOT NULL,
                overall_status TEXT NOT NULL,
                last_checked TEXT NOT NULL,
                policy_violations INTEGER DEFAULT 0,
                compliance_score REAL DEFAULT 0.0,
                details TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_default_policies(self):
        """Load default governance policies."""
        default_policies = [
            {
                "name": "PII Data Protection",
                "description": "Ensure PII data is properly protected and anonymized",
                "policy_type": PolicyType.DATA_PRIVACY,
                "severity": PolicySeverity.CRITICAL,
                "conditions": {
                    "has_pii": True,
                    "anonymization_required": True,
                    "encryption_required": True
                },
                "actions": [
                    {"type": "alert", "severity": "critical"},
                    {"type": "anonymize", "method": "k_anonymity"},
                    {"type": "encrypt", "algorithm": "AES256"}
                ],
                "tags": ["privacy", "gdpr", "ccpa"]
            },
            {
                "name": "Model Bias Detection",
                "description": "Monitor and prevent model bias across protected attributes",
                "policy_type": PolicyType.MODEL_BIAS,
                "severity": PolicySeverity.HIGH,
                "conditions": {
                    "bias_threshold": 0.1,
                    "protected_attributes": ["gender", "race", "age"],
                    "fairness_metrics": ["demographic_parity", "equalized_odds"]
                },
                "actions": [
                    {"type": "alert", "severity": "high"},
                    {"type": "retrain", "method": "fairness_constraint"},
                    {"type": "audit", "requirement": "bias_report"}
                ],
                "tags": ["fairness", "ethics", "bias"]
            },
            {
                "name": "Data Quality Standards",
                "description": "Enforce minimum data quality standards",
                "policy_type": PolicyType.DATA_QUALITY,
                "severity": PolicySeverity.MEDIUM,
                "conditions": {
                    "min_quality_score": 0.8,
                    "completeness_threshold": 0.95,
                    "consistency_threshold": 0.9
                },
                "actions": [
                    {"type": "alert", "severity": "medium"},
                    {"type": "quarantine", "duration_hours": 24},
                    {"type": "data_cleaning", "auto_fix": True}
                ],
                "tags": ["quality", "validation"]
            }
        ]
        
        for policy_data in default_policies:
            if not self.get_policy_by_name(policy_data["name"]):
                policy = PolicyRule(
                    id=self._generate_id(),
                    name=policy_data["name"],
                    description=policy_data["description"],
                    policy_type=policy_data["policy_type"],
                    severity=policy_data["severity"],
                    conditions=policy_data["conditions"],
                    actions=policy_data["actions"],
                    status=PolicyStatus.ACTIVE,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    created_by="system",
                    tags=policy_data["tags"]
                )
                self.create_policy(policy)
    
    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return hashlib.md5(f"{datetime.now().isoformat()}{threading.current_thread().ident}".encode()).hexdigest()[:16]
    
    def create_policy(self, policy: PolicyRule) -> str:
        """Create a new governance policy."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO policy_rules 
                (id, name, description, policy_type, severity, conditions, actions, 
                 status, created_at, updated_at, created_by, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                policy.id, policy.name, policy.description, policy.policy_type.value,
                policy.severity.value, json.dumps(policy.conditions), json.dumps(policy.actions),
                policy.status.value, policy.created_at.isoformat(), policy.updated_at.isoformat(),
                policy.created_by, json.dumps(policy.tags)
            ))
            
            conn.commit()
            conn.close()
            
            # Log audit entry
            self.log_audit_entry(
                event_type="policy_created",
                resource_id=policy.id,
                resource_type="policy",
                user_id=policy.created_by,
                action="create",
                details={"policy_name": policy.name, "policy_type": policy.policy_type.value}
            )
            
            return policy.id
    
    def update_policy(self, policy_id: str, updates: Dict[str, Any], user_id: str) -> bool:
        """Update an existing policy."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current policy
            cursor.execute('SELECT * FROM policy_rules WHERE id = ?', (policy_id,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return False
            
            # Update fields
            update_fields = []
            update_values = []
            
            for field, value in updates.items():
                if field in ['conditions', 'actions', 'tags']:
                    update_fields.append(f"{field} = ?")
                    update_values.append(json.dumps(value))
                elif field in ['policy_type', 'severity', 'status']:
                    update_fields.append(f"{field} = ?")
                    update_values.append(value.value if hasattr(value, 'value') else value)
                else:
                    update_fields.append(f"{field} = ?")
                    update_values.append(value)
            
            update_fields.append("updated_at = ?")
            update_values.append(datetime.now().isoformat())
            update_values.append(policy_id)
            
            cursor.execute(f'''
                UPDATE policy_rules 
                SET {", ".join(update_fields)}
                WHERE id = ?
            ''', update_values)
            
            conn.commit()
            conn.close()
            
            # Log audit entry
            self.log_audit_entry(
                event_type="policy_updated",
                resource_id=policy_id,
                resource_type="policy",
                user_id=user_id,
                action="update",
                details={"updates": updates}
            )
            
            return True
    
    def get_policy_by_name(self, name: str) -> Optional[PolicyRule]:
        """Get policy by name."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM policy_rules WHERE name = ?', (name,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return PolicyRule(
                id=row[0], name=row[1], description=row[2],
                policy_type=PolicyType(row[3]), severity=PolicySeverity(row[4]),
                conditions=json.loads(row[5]), actions=json.loads(row[6]),
                status=PolicyStatus(row[7]), created_at=datetime.fromisoformat(row[8]),
                updated_at=datetime.fromisoformat(row[9]), created_by=row[10],
                tags=json.loads(row[11]) if row[11] else []
            )
        return None
    
    def get_active_policies(self, policy_type: Optional[PolicyType] = None) -> List[PolicyRule]:
        """Get all active policies, optionally filtered by type."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if policy_type:
            cursor.execute(
                'SELECT * FROM policy_rules WHERE status = ? AND policy_type = ?',
                (PolicyStatus.ACTIVE.value, policy_type.value)
            )
        else:
            cursor.execute('SELECT * FROM policy_rules WHERE status = ?', (PolicyStatus.ACTIVE.value,))
        
        rows = cursor.fetchall()
        conn.close()
        
        policies = []
        for row in rows:
            policies.append(PolicyRule(
                id=row[0], name=row[1], description=row[2],
                policy_type=PolicyType(row[3]), severity=PolicySeverity(row[4]),
                conditions=json.loads(row[5]), actions=json.loads(row[6]),
                status=PolicyStatus(row[7]), created_at=datetime.fromisoformat(row[8]),
                updated_at=datetime.fromisoformat(row[9]), created_by=row[10],
                tags=json.loads(row[11]) if row[11] else []
            ))
        
        return policies
    
    def evaluate_compliance(self, resource_id: str, resource_type: str, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate compliance for a resource against all applicable policies."""
        policies = self.get_active_policies()
        compliance_results = {
            "resource_id": resource_id,
            "resource_type": resource_type,
            "overall_status": ComplianceStatus.COMPLIANT,
            "policy_results": [],
            "violations": [],
            "warnings": [],
            "score": 1.0,
            "timestamp": datetime.now()
        }
        
        total_policies = 0
        compliant_policies = 0
        
        for policy in policies:
            if self._policy_applies_to_resource(policy, resource_type):
                total_policies += 1
                result = self._evaluate_policy(policy, resource_data)
                compliance_results["policy_results"].append(result)
                
                if result["status"] == ComplianceStatus.COMPLIANT:
                    compliant_policies += 1
                elif result["status"] == ComplianceStatus.NON_COMPLIANT:
                    compliance_results["violations"].append(result)
                    if policy.severity in [PolicySeverity.HIGH, PolicySeverity.CRITICAL]:
                        compliance_results["overall_status"] = ComplianceStatus.NON_COMPLIANT
                elif result["status"] == ComplianceStatus.WARNING:
                    compliance_results["warnings"].append(result)
                    if compliance_results["overall_status"] == ComplianceStatus.COMPLIANT:
                        compliance_results["overall_status"] = ComplianceStatus.WARNING
        
        # Calculate compliance score
        if total_policies > 0:
            compliance_results["score"] = compliant_policies / total_policies
        
        # Store compliance event
        self._store_compliance_event(resource_id, resource_type, compliance_results)
        
        # Execute remediation actions if needed
        if compliance_results["violations"]:
            self._execute_remediation_actions(resource_id, resource_type, compliance_results["violations"])
        
        return compliance_results
    
    def _policy_applies_to_resource(self, policy: PolicyRule, resource_type: str) -> bool:
        """Check if a policy applies to a specific resource type."""
        # Basic implementation - can be extended with more sophisticated matching
        applicable_types = policy.conditions.get("applicable_resource_types", [])
        if not applicable_types:
            return True  # Policy applies to all resources if not specified
        return resource_type in applicable_types
    
    def _evaluate_policy(self, policy: PolicyRule, resource_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single policy against resource data."""
        result = {
            "policy_id": policy.id,
            "policy_name": policy.name,
            "status": ComplianceStatus.COMPLIANT,
            "details": {},
            "violations": [],
            "recommendations": []
        }
        
        try:
            # Evaluate conditions based on policy type
            if policy.policy_type == PolicyType.DATA_PRIVACY:
                result = self._evaluate_privacy_policy(policy, resource_data, result)
            elif policy.policy_type == PolicyType.MODEL_BIAS:
                result = self._evaluate_bias_policy(policy, resource_data, result)
            elif policy.policy_type == PolicyType.DATA_QUALITY:
                result = self._evaluate_quality_policy(policy, resource_data, result)
            elif policy.policy_type == PolicyType.SECURITY:
                result = self._evaluate_security_policy(policy, resource_data, result)
            else:
                result = self._evaluate_generic_policy(policy, resource_data, result)
            
        except Exception as e:
            self.logger.error(f"Error evaluating policy {policy.id}: {str(e)}")
            result["status"] = ComplianceStatus.UNKNOWN
            result["details"]["error"] = str(e)
        
        return result
    
    def _evaluate_privacy_policy(self, policy: PolicyRule, resource_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate privacy-related policy."""
        conditions = policy.conditions
        
        # Check for PII presence
        if conditions.get("has_pii") and resource_data.get("contains_pii", False):
            if not resource_data.get("is_anonymized", False) and conditions.get("anonymization_required", False):
                result["status"] = ComplianceStatus.NON_COMPLIANT
                result["violations"].append("PII data is not anonymized")
                result["recommendations"].append("Apply anonymization techniques")
            
            if not resource_data.get("is_encrypted", False) and conditions.get("encryption_required", False):
                result["status"] = ComplianceStatus.NON_COMPLIANT
                result["violations"].append("PII data is not encrypted")
                result["recommendations"].append("Enable encryption for PII data")
        
        return result
    
    def _evaluate_bias_policy(self, policy: PolicyRule, resource_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate bias-related policy."""
        conditions = policy.conditions
        bias_threshold = conditions.get("bias_threshold", 0.1)
        
        # Check bias metrics
        bias_metrics = resource_data.get("bias_metrics", {})
        for metric_name, metric_value in bias_metrics.items():
            if metric_value > bias_threshold:
                result["status"] = ComplianceStatus.NON_COMPLIANT
                result["violations"].append(f"Bias metric '{metric_name}' exceeds threshold: {metric_value} > {bias_threshold}")
                result["recommendations"].append(f"Retrain model with bias mitigation for {metric_name}")
        
        return result
    
    def _evaluate_quality_policy(self, policy: PolicyRule, resource_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate data quality policy."""
        conditions = policy.conditions
        
        quality_score = resource_data.get("quality_score", 1.0)
        min_quality = conditions.get("min_quality_score", 0.8)
        
        if quality_score < min_quality:
            result["status"] = ComplianceStatus.NON_COMPLIANT
            result["violations"].append(f"Quality score {quality_score} below minimum {min_quality}")
            result["recommendations"].append("Improve data quality through cleaning and validation")
        
        return result
    
    def _evaluate_security_policy(self, policy: PolicyRule, resource_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate security-related policy."""
        conditions = policy.conditions
        
        # Check encryption
        if conditions.get("encryption_required", False) and not resource_data.get("is_encrypted", False):
            result["status"] = ComplianceStatus.NON_COMPLIANT
            result["violations"].append("Encryption is required but not enabled")
            result["recommendations"].append("Enable encryption")
        
        # Check access controls
        if conditions.get("access_control_required", False) and not resource_data.get("has_access_control", False):
            result["status"] = ComplianceStatus.NON_COMPLIANT
            result["violations"].append("Access control is required but not configured")
            result["recommendations"].append("Configure proper access controls")
        
        return result
    
    def _evaluate_generic_policy(self, policy: PolicyRule, resource_data: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate generic policy conditions."""
        conditions = policy.conditions
        
        for condition_key, condition_value in conditions.items():
            if condition_key.startswith("_"):  # Skip internal conditions
                continue
            
            resource_value = resource_data.get(condition_key)
            
            if isinstance(condition_value, dict):
                # Handle complex conditions
                if "min" in condition_value and resource_value is not None:
                    if resource_value < condition_value["min"]:
                        result["status"] = ComplianceStatus.NON_COMPLIANT
                        result["violations"].append(f"{condition_key} value {resource_value} below minimum {condition_value['min']}")
                
                if "max" in condition_value and resource_value is not None:
                    if resource_value > condition_value["max"]:
                        result["status"] = ComplianceStatus.NON_COMPLIANT
                        result["violations"].append(f"{condition_key} value {resource_value} above maximum {condition_value['max']}")
            
            elif resource_value != condition_value:
                if condition_value is True and not resource_value:
                    result["status"] = ComplianceStatus.NON_COMPLIANT
                    result["violations"].append(f"{condition_key} is required but not present")
        
        return result
    
    def _store_compliance_event(self, resource_id: str, resource_type: str, compliance_results: Dict[str, Any]):
        """Store compliance evaluation results."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store individual policy results
            for policy_result in compliance_results["policy_results"]:
                event = ComplianceEvent(
                    id=self._generate_id(),
                    policy_id=policy_result["policy_id"],
                    resource_id=resource_id,
                    resource_type=resource_type,
                    status=policy_result["status"],
                    details=policy_result,
                    timestamp=compliance_results["timestamp"],
                    remediation_actions=policy_result.get("recommendations", [])
                )
                
                cursor.execute('''
                    INSERT INTO compliance_events 
                    (id, policy_id, resource_id, resource_type, status, details, timestamp, remediation_actions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.id, event.policy_id, event.resource_id, event.resource_type,
                    event.status.value, json.dumps(event.details), event.timestamp.isoformat(),
                    json.dumps(event.remediation_actions)
                ))
            
            # Update compliance summary
            cursor.execute('''
                INSERT OR REPLACE INTO compliance_summary 
                (resource_id, resource_type, overall_status, last_checked, policy_violations, compliance_score, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                resource_id, resource_type, compliance_results["overall_status"].value,
                compliance_results["timestamp"].isoformat(), len(compliance_results["violations"]),
                compliance_results["score"], json.dumps(compliance_results)
            ))
            
            conn.commit()
            conn.close()
    
    def _execute_remediation_actions(self, resource_id: str, resource_type: str, violations: List[Dict[str, Any]]):
        """Execute automated remediation actions for violations."""
        for violation in violations:
            policy_id = violation["policy_id"]
            policy = self.get_policy_by_id(policy_id)
            
            if policy:
                for action in policy.actions:
                    try:
                        self._execute_action(action, resource_id, resource_type, violation)
                    except Exception as e:
                        self.logger.error(f"Failed to execute remediation action: {str(e)}")
    
    def _execute_action(self, action: Dict[str, Any], resource_id: str, resource_type: str, violation: Dict[str, Any]):
        """Execute a specific remediation action."""
        action_type = action.get("type")
        
        if action_type == "alert":
            self._send_alert(action, resource_id, resource_type, violation)
        elif action_type == "quarantine":
            self._quarantine_resource(action, resource_id, resource_type)
        elif action_type == "anonymize":
            self._anonymize_data(action, resource_id, resource_type)
        elif action_type == "encrypt":
            self._encrypt_data(action, resource_id, resource_type)
        elif action_type == "retrain":
            self._trigger_retraining(action, resource_id, resource_type)
        elif action_type == "audit":
            self._trigger_audit(action, resource_id, resource_type, violation)
        
        # Log the action
        self.log_audit_entry(
            event_type="remediation_action",
            resource_id=resource_id,
            resource_type=resource_type,
            user_id="system",
            action=action_type,
            details={"action": action, "violation": violation}
        )
    
    def _send_alert(self, action: Dict[str, Any], resource_id: str, resource_type: str, violation: Dict[str, Any]):
        """Send compliance alert."""
        if self.mlops_integration:
            alert_data = {
                "type": "compliance_violation",
                "severity": action.get("severity", "medium"),
                "resource_id": resource_id,
                "resource_type": resource_type,
                "violation": violation,
                "timestamp": datetime.now().isoformat()
            }
            # Integrate with MLOps alerting system
            # self.mlops_integration.send_alert(alert_data)
    
    def _quarantine_resource(self, action: Dict[str, Any], resource_id: str, resource_type: str):
        """Quarantine a resource."""
        duration_hours = action.get("duration_hours", 24)
        # Implement resource quarantine logic
        self.logger.info(f"Quarantining {resource_type} {resource_id} for {duration_hours} hours")
    
    def _anonymize_data(self, action: Dict[str, Any], resource_id: str, resource_type: str):
        """Anonymize sensitive data."""
        method = action.get("method", "k_anonymity")
        # Implement data anonymization
        self.logger.info(f"Anonymizing {resource_type} {resource_id} using {method}")
    
    def _encrypt_data(self, action: Dict[str, Any], resource_id: str, resource_type: str):
        """Encrypt sensitive data."""
        algorithm = action.get("algorithm", "AES256")
        # Implement data encryption
        self.logger.info(f"Encrypting {resource_type} {resource_id} using {algorithm}")
    
    def _trigger_retraining(self, action: Dict[str, Any], resource_id: str, resource_type: str):
        """Trigger model retraining."""
        method = action.get("method", "standard")
        # Integrate with model training pipeline
        self.logger.info(f"Triggering retraining for {resource_type} {resource_id} using {method}")
    
    def _trigger_audit(self, action: Dict[str, Any], resource_id: str, resource_type: str, violation: Dict[str, Any]):
        """Trigger compliance audit."""
        requirement = action.get("requirement", "standard_audit")
        # Implement audit triggering
        self.logger.info(f"Triggering audit for {resource_type} {resource_id} - requirement: {requirement}")
    
    def get_policy_by_id(self, policy_id: str) -> Optional[PolicyRule]:
        """Get policy by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM policy_rules WHERE id = ?', (policy_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return PolicyRule(
                id=row[0], name=row[1], description=row[2],
                policy_type=PolicyType(row[3]), severity=PolicySeverity(row[4]),
                conditions=json.loads(row[5]), actions=json.loads(row[6]),
                status=PolicyStatus(row[7]), created_at=datetime.fromisoformat(row[8]),
                updated_at=datetime.fromisoformat(row[9]), created_by=row[10],
                tags=json.loads(row[11]) if row[11] else []
            )
        return None
    
    def log_audit_entry(self, event_type: str, resource_id: str, resource_type: str, 
                       user_id: str, action: str, details: Dict[str, Any],
                       ip_address: Optional[str] = None, session_id: Optional[str] = None):
        """Log an audit entry."""
        with self._lock:
            entry = AuditEntry(
                id=self._generate_id(),
                event_type=event_type,
                resource_id=resource_id,
                resource_type=resource_type,
                user_id=user_id,
                action=action,
                details=details,
                timestamp=datetime.now(),
                ip_address=ip_address,
                session_id=session_id
            )
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO audit_trail 
                (id, event_type, resource_id, resource_type, user_id, action, details, timestamp, ip_address, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.id, entry.event_type, entry.resource_id, entry.resource_type,
                entry.user_id, entry.action, json.dumps(entry.details),
                entry.timestamp.isoformat(), entry.ip_address, entry.session_id
            ))
            
            conn.commit()
            conn.close()
    
    def get_compliance_report(self, resource_id: Optional[str] = None, 
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate compliance report."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query conditions
        conditions = []
        params = []
        
        if resource_id:
            conditions.append("resource_id = ?")
            params.append(resource_id)
        
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat())
        
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date.isoformat())
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Get compliance events
        cursor.execute(f'''
            SELECT policy_id, status, COUNT(*) as count
            FROM compliance_events 
            {where_clause}
            GROUP BY policy_id, status
        ''', params)
        
        compliance_stats = cursor.fetchall()
        
        # Get audit trail
        cursor.execute(f'''
            SELECT event_type, COUNT(*) as count
            FROM audit_trail 
            {where_clause}
            GROUP BY event_type
        ''', params)
        
        audit_stats = cursor.fetchall()
        
        # Get compliance summary
        if resource_id:
            cursor.execute('SELECT * FROM compliance_summary WHERE resource_id = ?', (resource_id,))
        else:
            cursor.execute('SELECT * FROM compliance_summary')
        
        summary_data = cursor.fetchall()
        
        conn.close()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            },
            "resource_id": resource_id,
            "compliance_statistics": {
                "by_policy_status": {f"{row[0]}_{row[1]}": row[2] for row in compliance_stats}
            },
            "audit_statistics": {
                "by_event_type": {row[0]: row[1] for row in audit_stats}
            },
            "compliance_summary": [
                {
                    "resource_id": row[0],
                    "resource_type": row[1],
                    "overall_status": row[2],
                    "last_checked": row[3],
                    "violations": row[4],
                    "score": row[5]
                } for row in summary_data
            ]
        }
        
        return report
    
    def start_scheduled_compliance_checks(self, check_interval_hours: int = 24):
        """Start scheduled compliance monitoring."""
        def run_scheduler():
            while self._running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        def scheduled_compliance_check():
            """Run compliance checks for all resources."""
            try:
                # This would integrate with your data catalog or resource registry
                # For now, we'll demonstrate with sample resources
                self.logger.info("Running scheduled compliance check")
                
                # Get resources from data catalog integration
                if hasattr(self, 'data_catalog_integration') and self.data_catalog_integration:
                    resources = self.data_catalog_integration.get_all_resources()
                    for resource in resources:
                        self.evaluate_compliance(
                            resource['id'],
                            resource['type'],
                            resource['metadata']
                        )
                
            except Exception as e:
                self.logger.error(f"Scheduled compliance check failed: {str(e)}")
        
        schedule.every(check_interval_hours).hours.do(scheduled_compliance_check)
        
        self._running = True
        self._scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self._scheduler_thread.start()
        
        self.logger.info(f"Started scheduled compliance checks every {check_interval_hours} hours")
    
    def stop_scheduled_compliance_checks(self):
        """Stop scheduled compliance monitoring."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        schedule.clear()
        self.logger.info("Stopped scheduled compliance checks")

# Example usage
if __name__ == "__main__":
    # Initialize governance automation
    governance = GovernanceAutomationEngine()
    
    # Example resource data
    sample_resource = {
        "id": "dataset_001",
        "type": "dataset",
        "contains_pii": True,
        "is_anonymized": False,
        "is_encrypted": True,
        "quality_score": 0.85,
        "bias_metrics": {
            "demographic_parity": 0.05,
            "equalized_odds": 0.12
        }
    }
    
    # Evaluate compliance
    results = governance.evaluate_compliance(
        "dataset_001",
        "dataset", 
        sample_resource
    )
    
    print("Compliance Results:")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Compliance Score: {results['score']:.2f}")
    print(f"Violations: {len(results['violations'])}")
    print(f"Warnings: {len(results['warnings'])}")
    
    # Generate compliance report
    report = governance.get_compliance_report()
    print(f"\nCompliance Report Generated at: {report['generated_at']}")
    
    # Start scheduled monitoring
    governance.start_scheduled_compliance_checks(check_interval_hours=1)
    
    print("Governance automation system initialized and running!")
