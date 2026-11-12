# complaint_classifier_app.py - Automated Complaint Management System
# With Dynamic Severity Escalation & Manual Override

import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import os
import re
import hashlib
from datetime import datetime, timedelta
import time
import json
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set HF_TOKEN for accessing gated models
HF_TOKEN = os.getenv("HF_TOKEN")

# Set page config
st.set_page_config(
    page_title="CivicRedress-Automated Complaint Management System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR STYLING (Redesigned)
# ============================================================================

def load_css():
    st.markdown("""
    <style>
    /* Global */
    .main {
        padding: 0rem 1rem;
        background: #f5f7fa;
    }
    
    /* Header */
    .header-container {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .header-container h1 {
        color: white;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.4);
    }
    .header-container p {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        letter-spacing: 0.3px;
    }
    
    /* Card */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
        border-left: 4px solid #2a5298;
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.3rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }
    .result-card .label {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .result-card .value {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .result-card .sub {
        font-size: 0.85rem;
        color: #666;
    }
    
    /* Severity comparison boxes */
    .severity-box {
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        height: 100%;
    }
    .severity-box .label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.85;
        margin-bottom: 0.4rem;
    }
    .severity-box .value {
        font-size: 1.8rem;
        font-weight: 900;
        margin: 0.3rem 0;
    }
    .severity-box .note {
        font-size: 0.8rem;
        opacity: 0.9;
        margin-top: 0.4rem;
    }
    .severity-model {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        color: #0d47a1;
        border: 2px solid #1976d2;
    }
    .severity-current {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        color: #e65100;
        border: 2px solid #f57c00;
    }
    .severity-critical {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        color: #b71c1c;
        border: 2px solid #c62828;
        animation: pulse-red 2s infinite;
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 0 0 rgba(198, 40, 40, 0.4); }
        50% { box-shadow: 0 0 0 12px rgba(198, 40, 40, 0); }
    }
    
    /* Big red alert for critical overdue */
    .critical-alert {
        background: linear-gradient(135deg, #c62828 0%, #b71c1c 100%);
        color: #ffffff !important;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 800;
        letter-spacing: 1px;
        box-shadow: 0 8px 24px rgba(198, 40, 40, 0.5);
        animation: pulse-red 2s infinite;
        margin: 1rem 0;
    }
    .critical-alert * {
        color: #ffffff !important;
    }
    
    /* Data in red for critical */
    .critical-data {
        color: #c62828 !important;
        font-weight: 800 !important;
        font-size: 1.1rem;
    }
    .overdue-data {
        color: #d32f2f !important;
        font-weight: 700 !important;
        background: #ffebee;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
    }
    .escalated-data {
        color: #d32f2f !important;
        font-weight: 800 !important;
        font-size: 1.15rem;
        background: #ffcdd2;
        padding: 0.2rem 0.6rem;
        border-radius: 8px;
    }
    
    /* Severity badges */
    .sev-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .sev-low { background: #d4edda; color: #155724; }
    .sev-medium { background: #fff3cd; color: #856404; }
    .sev-high { background: #ffe0b2; color: #e65100; }
    .sev-critical { background: #f8d7da; color: #721c24; border: 1px solid #c62828; }
    
    /* Priority badges */
    .priority-critical { background: #dc3545; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
    .priority-high { background: #fd7e14; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
    .priority-medium { background: #ffc107; color: #212529; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
    .priority-low { background: #28a745; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
    
    /* Status */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .status-resolved { background: #d4edda; color: #155724; }
    .status-pending { background: #fff3cd; color: #856404; }
    .status-in-progress { background: #cce5ff; color: #004085; }
    .status-escalated { background: #f8d7da; color: #721c24; }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        text-align: center;
        border-bottom: 3px solid #2a5298;
    }
    .metric-card .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1e3c72;
    }
    .metric-card .metric-label {
        font-size: 0.7rem;
        color: #666;
        font-weight: 600;
    }
    
    /* Timeline */
    .timeline-item {
        padding: 0.6rem 1rem;
        border-left: 3px solid #2a5298;
        margin: 0.4rem 0;
        background: #f8f9fa;
        border-radius: 0 8px 8px 0;
    }
    .timeline-item .time { font-size: 0.78rem; color: #666; }
    .timeline-item .event { font-weight: 600; color: #1e3c72; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        font-weight: 600;
        padding: 0.55rem 1.5rem;
        border-radius: 25px;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(42, 82, 152, 0.4);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 2rem;
        background: #f8f9fa;
        border-radius: 12px;
        border-top: 3px solid #2a5298;
    }
    .footer .creator { font-weight: 600; color: #1e3c72; }
    .footer .guide { color: #555; font-style: italic; }
    
    /* Red large text for critical warnings */
    .red-critical-text {
        color: #c62828 !important;
        font-weight: 900 !important;
        font-size: 1.15em;
    }
    </style>
    """, unsafe_allow_html=True)



# ============================================================================
# DYNAMIC SEVERITY ESCALATION ENGINE
# ============================================================================

def compute_current_severity(model_severity, complaint_datetime_str, status, manual_severity=None):
    """
    Compute the current (effective) severity based on time-based escalation rules.
    
    Rules:
      - If manual override exists → use manual
      - Else if status is Resolved → keep model severity
      - Else apply time-based escalation:
          * Model = Low:
              - < 7 days  → Low
              - 7-14 days → Medium
              - >= 14 days → High
          * Model = Medium:
              - < 3 days  → Medium
              - 3-7 days  → High
              - >= 7 days → Critical
          * Model = High:
              - < 3 days  → High
              - >= 3 days → Critical
          * Model = Critical → always Critical
    
    Returns dict with:
      - current_severity
      - model_severity
      - escalation_reason (str or None)
      - days_old
      - is_overdue (bool)
      - is_critical_overdue (bool)
    """
    # Parse complaint datetime
    days_old = 0.0
    try:
        created = datetime.strptime(str(complaint_datetime_str), "%Y-%m-%d %H:%M:%S")
        delta = datetime.now() - created
        days_old = delta.total_seconds() / 86400
    except Exception:
        days_old = 0.0
    
    # If manually overridden, use manual severity directly
    if manual_severity and str(manual_severity).strip():
        manual = str(manual_severity).strip().title()
        return {
            'model_severity': model_severity,
            'current_severity': manual,
            'escalation_reason': f"Manually assigned by officer",
            'days_old': round(days_old, 2),
            'is_overdue': False,
            'is_critical_overdue': manual == 'Critical',
            'is_manual': True,
        }
    
    # Resolved complaints: keep model severity (no escalation)
    if status == 'Resolved':
        return {
            'model_severity': model_severity,
            'current_severity': model_severity,
            'escalation_reason': None,
            'days_old': round(days_old, 2),
            'is_overdue': False,
            'is_critical_overdue': False,
            'is_manual': False,
        }
    
    # Determine current severity
    sev = str(model_severity).strip().title()
    reason = None
    current = sev
    
    if sev == 'Low':
        if days_old >= 14:
            current = 'High'
            reason = f"Low → High: unresolved for {days_old:.1f} days (≥14 days)"
        elif days_old >= 7:
            current = 'Medium'
            reason = f"Low → Medium: unresolved for {days_old:.1f} days (≥7 days)"
    elif sev == 'Medium':
        if days_old >= 7:
            current = 'Critical'
            reason = f"Medium → Critical: unresolved for {days_old:.1f} days (≥7 days)"
        elif days_old >= 3:
            current = 'High'
            reason = f"Medium → High: unresolved for {days_old:.1f} days (≥3 days)"
    elif sev == 'High':
        if days_old >= 3:
            current = 'Critical'
            reason = f"High → Critical: unresolved for {days_old:.1f} days (≥3 days)"
    elif sev == 'Critical':
        current = 'Critical'
    
    is_critical_overdue = (current == 'Critical' and days_old >= 3)
    
    return {
        'model_severity': sev,
        'current_severity': current,
        'escalation_reason': reason,
        'days_old': round(days_old, 2),
        'is_overdue': current != sev,
        'is_critical_overdue': is_critical_overdue,
        'is_manual': False,
    }


def severity_order(sev):
    """Numeric order for severity comparison"""
    order = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    return order.get(str(sev).title(), 2)


def severity_color_class(sev):
    """Return CSS class for a severity"""
    s = str(sev).title()
    return {
        'Low': 'sev-low',
        'Medium': 'sev-medium',
        'High': 'sev-high',
        'Critical': 'sev-critical',
    }.get(s, 'sev-medium')


def severity_emoji(sev):
    return {
        'Low': '🟢',
        'Medium': '🟡',
        'High': '🟠',
        'Critical': '🔴',
    }.get(str(sev).title(), '🟡')


# ============================================================================
# ENHANCED COMPLAINT STORE
# ============================================================================

class ComplaintStore:
    """Store and manage complaints with automated workflow, dynamic severity, and SLA tracking"""
    
    SLA_CONFIG = {
        'Low':      {'response': 24, 'resolution': 168},
        'Medium':   {'response': 12, 'resolution': 72},
        'High':     {'response': 4,  'resolution': 24},
        'Critical': {'response': 1,  'resolution': 8},
    }
    
    DEPARTMENT_MAP = {
        'Road & Transportation': 'Public Works Department',
        'Water Leakage': 'Water Supply Department',
        'Electricity': 'Electricity Board',
        'Waste Management': 'Sanitation Department',
        'Other': 'General Administration'
    }
    
    ASSIGNEE_MAP = {
        'Road & Transportation': 'Road Maintenance Team',
        'Water Leakage': 'Water Works Team',
        'Electricity': 'Electrical Maintenance Team',
        'Waste Management': 'Sanitation Team',
        'Other': 'General Team'
    }
    
    SCHEMA_VERSION = 3
    
    def __init__(self, storage_file='complaints_data.json'):
        self.storage_file = storage_file
        self.complaints = []
        self.load_data()
    
    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------
    def _get_default_complaint(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            'id': '',
            'name': '',
            'location': '',
            'datetime': now,
            'title': '',
            'description': '',
            'primary_category': 'Other',
            'subcategory': 'Other',
            'model_severity': 'Medium',       # Original model prediction (never overwritten)
            'manual_severity': None,          # Officer override (if any)
            'severity': 'Medium',             # Effective/current severity (denormalized)
            'priority': 'Medium',
            'action_taken': 'Awaiting action',
            'status': 'Pending',
            'resolution_date': None,
            'assigned_to': 'Not assigned',
            'department': 'General Administration',
            'sla_deadline': None,
            'response_time': None,
            'resolution_time': None,
            'sla_met': None,
            'escalated': False,
            'escalation_reason': None,
            'timeline': [],
            'feedback_rating': None,
            'feedback_comments': None,
            'schema_version': self.SCHEMA_VERSION,
        }
    
    def _migrate_complaint(self, c):
        default = self._get_default_complaint()
        migrated = False
        
        # Handle legacy 'severity' → 'model_severity'
        if 'model_severity' not in c:
            c['model_severity'] = c.get('severity', 'Medium')
            migrated = True
        
        if 'manual_severity' not in c:
            c['manual_severity'] = None
            migrated = True
        
        # Fill missing fields
        for key, default_value in default.items():
            if key not in c:
                c[key] = default_value
                migrated = True
        
        # Recompute current severity
        computed = compute_current_severity(
            c.get('model_severity', 'Medium'),
            c.get('datetime', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            c.get('status', 'Pending'),
            c.get('manual_severity')
        )
        if c.get('severity') != computed['current_severity']:
            c['severity'] = computed['current_severity']
            migrated = True
        
        # Recompute priority from current severity
        correct_priority = self.calculate_priority(
            computed['current_severity'],
            c.get('primary_category', 'Other'),
            c.get('description', '')
        )
        if c.get('priority') != correct_priority:
            c['priority'] = correct_priority
            migrated = True
        
        # Department / assignee
        if not c.get('department') or c['department'] == 'General Administration':
            c['department'] = self.get_department(c.get('primary_category', 'Other'))
            migrated = True
        if not c.get('assigned_to') or c['assigned_to'] in ('Not assigned', 'Unassigned', ''):
            c['assigned_to'] = self.get_assignee(c.get('primary_category', 'Other'))
            migrated = True
        
        # SLA
        if not c.get('sla_deadline'):
            c['sla_deadline'] = self.calculate_sla_deadline(
                c.get('severity', 'Medium'),
                c.get('datetime', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            migrated = True
        
        # Response time
        if c.get('response_time') is None and c.get('status') != 'Pending':
            try:
                created = datetime.strptime(c['datetime'], "%Y-%m-%d %H:%M:%S")
                if c.get('timeline') and len(c['timeline']) > 1:
                    responded = datetime.strptime(c['timeline'][1]['time'], "%Y-%m-%d %H:%M:%S")
                    c['response_time'] = round((responded - created).total_seconds() / 3600, 1)
                    migrated = True
            except Exception:
                pass
        
        # Resolution time & SLA met
        if c.get('resolution_time') is None and c.get('status') == 'Resolved' and c.get('resolution_date'):
            try:
                created = datetime.strptime(c['datetime'], "%Y-%m-%d %H:%M:%S")
                resolved = datetime.strptime(c['resolution_date'], "%Y-%m-%d %H:%M:%S")
                c['resolution_time'] = round((resolved - created).total_seconds() / 3600, 1)
                migrated = True
            except Exception:
                pass
        
        if c.get('sla_met') is None and c.get('status') == 'Resolved' and c.get('resolution_date') and c.get('sla_deadline'):
            try:
                resolved = datetime.strptime(c['resolution_date'], "%Y-%m-%d %H:%M:%S")
                deadline = datetime.strptime(c['sla_deadline'], "%Y-%m-%d %H:%M:%S")
                c['sla_met'] = resolved <= deadline
                migrated = True
            except Exception:
                pass
        
        # Timeline
        if not c.get('timeline'):
            timeline = []
            if c.get('datetime'):
                timeline.append({'time': c['datetime'], 'event': 'Complaint registered'})
            if c.get('assigned_to') and c['assigned_to'] not in ('Not assigned', 'Unassigned'):
                timeline.append({
                    'time': c.get('datetime', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    'event': f"Assigned to {c['assigned_to']}"
                })
            if c.get('status') == 'Resolved' and c.get('resolution_date'):
                timeline.append({
                    'time': c['resolution_date'],
                    'event': f"Resolved - {c.get('action_taken', 'Completed')}"
                })
            c['timeline'] = timeline
            migrated = True
        
        if c.get('schema_version') != self.SCHEMA_VERSION:
            c['schema_version'] = self.SCHEMA_VERSION
            migrated = True
        
        return migrated
    
    def _migrate_data(self):
        count = 0
        for c in self.complaints:
            if self._migrate_complaint(c):
                count += 1
        if count > 0:
            self.save_data()
            print(f"✅ Migrated {count} complaint(s) to schema v{self.SCHEMA_VERSION}")
    
    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------
    def load_data(self):
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                if isinstance(raw, dict) and 'complaints' in raw:
                    self.complaints = raw['complaints']
                elif isinstance(raw, list):
                    self.complaints = raw
                else:
                    self.complaints = []
                self._migrate_data()
            else:
                self.complaints = self._create_sample_data()
                self.save_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            self.complaints = self._create_sample_data()
            self.save_data()
    
    def save_data(self):
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.complaints, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def refresh_severities(self):
        """Recompute current severity for all unresolved complaints based on time."""
        changed = 0
        for c in self.complaints:
            computed = compute_current_severity(
                c.get('model_severity', 'Medium'),
                c.get('datetime'),
                c.get('status', 'Pending'),
                c.get('manual_severity')
            )
            if c.get('severity') != computed['current_severity']:
                c['severity'] = computed['current_severity']
                # Recompute priority
                c['priority'] = self.calculate_priority(
                    computed['current_severity'],
                    c.get('primary_category', 'Other'),
                    c.get('description', '')
                )
                # Update SLA deadline based on current severity
                c['sla_deadline'] = self.calculate_sla_deadline(
                    computed['current_severity'],
                    c.get('datetime')
                )
                # Log timeline event
                if 'timeline' not in c or not isinstance(c['timeline'], list):
                    c['timeline'] = []
                if computed['escalation_reason']:
                    c['timeline'].append({
                        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'event': f"⬆️ Severity escalated: {computed['escalation_reason']}"
                    })
                changed += 1
        
        if changed > 0:
            self.save_data()
        return changed
    
    # ------------------------------------------------------------------
    # Business logic
    # ------------------------------------------------------------------
    def calculate_priority(self, severity, category, description):
        priority_map = {'High': 'High', 'Medium': 'Medium', 'Low': 'Low', 'Critical': 'Critical'}
        priority = priority_map.get(str(severity).title(), 'Medium')
        
        critical_categories = ['Water Leakage', 'Electricity']
        if category in critical_categories and str(severity).title() in ('High', 'Critical'):
            priority = 'Critical'
        
        if description:
            emergency_keywords = ['burst', 'fire', 'electric shock', 'collapse',
                                  'emergency', 'danger', 'hazard', 'explosion']
            if any(kw in description.lower() for kw in emergency_keywords):
                priority = 'Critical'
        
        return priority
    
    def calculate_sla_deadline(self, severity, complaint_datetime):
        sla_hours = self.SLA_CONFIG.get(str(severity).title(), {}).get('resolution', 72)
        try:
            complaint_dt = datetime.strptime(str(complaint_datetime), "%Y-%m-%d %H:%M:%S")
        except Exception:
            complaint_dt = datetime.now()
        deadline = complaint_dt + timedelta(hours=sla_hours)
        return deadline.strftime("%Y-%m-%d %H:%M:%S")
    
    def get_department(self, category):
        return self.DEPARTMENT_MAP.get(category, 'General Administration')
    
    def get_assignee(self, category):
        return self.ASSIGNEE_MAP.get(category, 'General Team')
    
    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add_complaint(self, complaint_data):
        max_id = 0
        for c in self.complaints:
            try:
                num = int(c['id'].replace('CMP', ''))
                if num > max_id:
                    max_id = num
            except Exception:
                pass
        complaint_data['id'] = f"CMP{max_id + 1:03d}"
        
        # Original model severity — never overwritten
        model_sev = str(complaint_data.get('severity', 'Medium')).title()
        complaint_data['model_severity'] = model_sev
        complaint_data['manual_severity'] = None
        
        # Current (effective) severity = model severity at creation
        complaint_data['severity'] = model_sev
        
        # Priority
        complaint_data['priority'] = self.calculate_priority(
            model_sev,
            complaint_data.get('primary_category', 'Other'),
            complaint_data.get('description', '')
        )
        
        # Department / assignee
        category = complaint_data.get('primary_category', 'Other')
        complaint_data['department'] = self.get_department(category)
        complaint_data['assigned_to'] = self.get_assignee(category)
        
        # Status and workflow
        complaint_data['status'] = 'Pending'
        complaint_data['action_taken'] = 'Auto-assigned to ' + complaint_data['assigned_to']
        complaint_data['resolution_date'] = None
        complaint_data['sla_deadline'] = self.calculate_sla_deadline(
            model_sev,
            complaint_data.get('datetime', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        complaint_data['response_time'] = None
        complaint_data['resolution_time'] = None
        complaint_data['sla_met'] = None
        complaint_data['escalated'] = False
        complaint_data['escalation_reason'] = None
        complaint_data['feedback_rating'] = None
        complaint_data['feedback_comments'] = None
        complaint_data['schema_version'] = self.SCHEMA_VERSION
        
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        complaint_data['timeline'] = [
            {'time': complaint_data.get('datetime', now_str),
             'event': 'Complaint registered'},
            {'time': now_str,
             'event': f"AI predicted severity: {model_sev}"},
            {'time': now_str,
             'event': f"Auto-assigned to {complaint_data['assigned_to']} ({complaint_data['department']})"}
        ]
        
        self.complaints.append(complaint_data)
        self.save_data()
        return complaint_data['id']
    
    def update_status(self, complaint_id, status, action_taken=None, assigned_to=None):
        for c in self.complaints:
            if c['id'] == complaint_id:
                old_status = c['status']
                c['status'] = status
                if action_taken:
                    c['action_taken'] = action_taken
                if assigned_to:
                    c['assigned_to'] = assigned_to
                
                if 'timeline' not in c or not isinstance(c['timeline'], list):
                    c['timeline'] = []
                
                event = f"Status: {old_status} → {status}"
                if action_taken:
                    event += f" | {action_taken}"
                c['timeline'].append({
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'event': event
                })
                
                if status == 'Resolved':
                    c['resolution_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    try:
                        created = datetime.strptime(c['datetime'], "%Y-%m-%d %H:%M:%S")
                        resolved = datetime.strptime(c['resolution_date'], "%Y-%m-%d %H:%M:%S")
                        c['resolution_time'] = round((resolved - created).total_seconds() / 3600, 1)
                    except Exception:
                        c['resolution_time'] = None
                    if c.get('sla_deadline'):
                        try:
                            deadline = datetime.strptime(c['sla_deadline'], "%Y-%m-%d %H:%M:%S")
                            c['sla_met'] = resolved <= deadline
                        except Exception:
                            c['sla_met'] = None
                
                if old_status == 'Pending' and status != 'Pending' and c.get('response_time') is None:
                    try:
                        created = datetime.strptime(c['datetime'], "%Y-%m-%d %H:%M:%S")
                        c['response_time'] = round((datetime.now() - created).total_seconds() / 3600, 1)
                    except Exception:
                        c['response_time'] = None
                
                # Recompute current severity on status change
                computed = compute_current_severity(
                    c.get('model_severity', 'Medium'),
                    c.get('datetime'),
                    status,
                    c.get('manual_severity')
                )
                c['severity'] = computed['current_severity']
                c['priority'] = self.calculate_priority(
                    computed['current_severity'],
                    c.get('primary_category', 'Other'),
                    c.get('description', '')
                )
                
                self.save_data()
                return True
        return False
    
    def override_severity(self, complaint_id, new_severity, reason=None):
        """
        Officer manually overrides the severity.
        Preserves model_severity for future model retraining.
        """
        for c in self.complaints:
            if c['id'] == complaint_id:
                old_current = c.get('severity', 'Medium')
                new_severity = str(new_severity).title()
                
                c['manual_severity'] = new_severity
                c['severity'] = new_severity
                
                # Update priority based on manual severity
                c['priority'] = self.calculate_priority(
                    new_severity,
                    c.get('primary_category', 'Other'),
                    c.get('description', '')
                )
                
                # Recompute SLA deadline based on new severity
                c['sla_deadline'] = self.calculate_sla_deadline(
                    new_severity,
                    c.get('datetime')
                )
                
                if 'timeline' not in c or not isinstance(c['timeline'], list):
                    c['timeline'] = []
                
                reason_txt = f" | Reason: {reason}" if reason else ""
                c['timeline'].append({
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'event': f"✏️ Officer override: {old_current} → {new_severity}{reason_txt}"
                })
                
                self.save_data()
                return True
        return False
    
    def clear_severity_override(self, complaint_id):
        """Remove manual override and restore time-based escalation."""
        for c in self.complaints:
            if c['id'] == complaint_id:
                c['manual_severity'] = None
                computed = compute_current_severity(
                    c.get('model_severity', 'Medium'),
                    c.get('datetime'),
                    c.get('status', 'Pending'),
                    None
                )
                c['severity'] = computed['current_severity']
                c['priority'] = self.calculate_priority(
                    computed['current_severity'],
                    c.get('primary_category', 'Other'),
                    c.get('description', '')
                )
                
                if 'timeline' not in c or not isinstance(c['timeline'], list):
                    c['timeline'] = []
                c['timeline'].append({
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'event': f"↩️ Manual override removed — reverted to auto severity ({computed['current_severity']})"
                })
                
                self.save_data()
                return True
        return False
    
    def escalate_complaint(self, complaint_id, reason):
        for c in self.complaints:
            if c['id'] == complaint_id:
                c['escalated'] = True
                c['escalation_reason'] = reason
                c['priority'] = 'Critical'
                c['status'] = 'Escalated'
                
                if 'timeline' not in c or not isinstance(c['timeline'], list):
                    c['timeline'] = []
                c['timeline'].append({
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'event': f"🚨 ESCALATED: {reason}"
                })
                self.save_data()
                return True
        return False
    
    def add_feedback(self, complaint_id, rating, comments=None):
        for c in self.complaints:
            if c['id'] == complaint_id:
                c['feedback_rating'] = rating
                c['feedback_comments'] = comments
                if 'timeline' not in c or not isinstance(c['timeline'], list):
                    c['timeline'] = []
                c['timeline'].append({
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'event': f"Feedback: {rating}/5 stars"
                })
                self.save_data()
                return True
        return False
    
    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_all_complaints(self):
        if not self.complaints:
            return pd.DataFrame(columns=list(self._get_default_complaint().keys()))
        df = pd.DataFrame(self.complaints)
        for col in self._get_default_complaint().keys():
            if col not in df.columns:
                df[col] = None
        return df
    
    def get_statistics(self):
        df = self.get_all_complaints()
        if df.empty:
            return {
                'total': 0, 'resolved': 0, 'pending': 0, 'in_progress': 0,
                'escalated': 0, 'critical': 0, 'high': 0, 'medium': 0, 'low': 0,
                'auto_escalated': 0, 'manual_overrides': 0,
                'categories': {}, 'departments': {},
                'avg_resolution_time': 0, 'avg_response_time': 0,
                'sla_compliance': 0, 'avg_rating': 0,
                'resolved_today': 0, 'pending_overdue': 0
            }
        
        resolved_df = df[df['status'] == 'Resolved']
        sla_met = 0
        sla_total = 0
        if 'sla_met' in resolved_df.columns:
            valid = resolved_df[resolved_df['sla_met'].notna()]
            sla_met = int(valid['sla_met'].sum()) if not valid.empty else 0
            sla_total = len(valid)
        
        avg_resolution = 0
        if 'resolution_time' in resolved_df.columns and not resolved_df.empty:
            v = resolved_df['resolution_time'].dropna()
            avg_resolution = v.mean() if not v.empty else 0
        
        avg_response = 0
        if 'response_time' in df.columns:
            v = df['response_time'].dropna()
            avg_response = v.mean() if not v.empty else 0
        
        avg_rating = 0
        if 'feedback_rating' in df.columns:
            v = df['feedback_rating'].dropna()
            avg_rating = v.mean() if not v.empty else 0
        
        today = datetime.now().strftime("%Y-%m-%d")
        resolved_today = 0
        if not resolved_df.empty and 'resolution_date' in resolved_df.columns:
            resolved_today = len(resolved_df[
                resolved_df['resolution_date'].astype(str).str.startswith(today, na=False)
            ])
        
        overdue = 0
        for _, row in df.iterrows():
            if row['status'] in ['Pending', 'In Progress', 'Escalated', 'Assigned'] and row.get('sla_deadline'):
                try:
                    dl = datetime.strptime(str(row['sla_deadline']), "%Y-%m-%d %H:%M:%S")
                    if datetime.now() > dl:
                        overdue += 1
                except Exception:
                    pass
        
        escalated_count = int(df['escalated'].fillna(False).astype(bool).sum()) if 'escalated' in df.columns else 0
        critical_count = int((df['severity'] == 'Critical').sum()) if 'severity' in df.columns else 0
        
        auto_escalated = 0
        manual_overrides = 0
        for c in self.complaints:
            ms = c.get('model_severity')
            cs = c.get('severity')
            man = c.get('manual_severity')
            if man:
                manual_overrides += 1
            elif ms and cs and ms != cs:
                auto_escalated += 1
        
        return {
            'total': len(df),
            'resolved': len(df[df['status'] == 'Resolved']),
            'pending': len(df[df['status'] == 'Pending']),
            'in_progress': len(df[df['status'] == 'In Progress']),
            'escalated': escalated_count,
            'critical': critical_count,
            'high': len(df[df['severity'] == 'High']),
            'medium': len(df[df['severity'] == 'Medium']),
            'low': len(df[df['severity'] == 'Low']),
            'auto_escalated': auto_escalated,
            'manual_overrides': manual_overrides,
            'categories': df['primary_category'].value_counts().to_dict(),
            'departments': df['department'].value_counts().to_dict(),
            'avg_resolution_time': round(avg_resolution, 1) if avg_resolution else 0,
            'avg_response_time': round(avg_response, 1) if avg_response else 0,
            'sla_compliance': round(sla_met / sla_total * 100, 1) if sla_total > 0 else 0,
            'avg_rating': round(avg_rating, 1) if avg_rating else 0,
            'resolved_today': resolved_today,
            'pending_overdue': overdue
        }
    
    def get_trends(self, days=7):
        df = self.get_all_complaints()
        if df.empty:
            return pd.DataFrame()
        try:
            df = df.copy()
            df['date'] = pd.to_datetime(df['datetime'], errors='coerce').dt.date
            df = df.dropna(subset=['date'])
            cutoff = datetime.now().date() - timedelta(days=days)
            df = df[df['date'] >= cutoff]
            if df.empty:
                return pd.DataFrame()
            trends = df.groupby('date').agg(
                total=('id', 'count'),
                resolved=('status', lambda x: (x == 'Resolved').sum())
            )
            trends['pending'] = trends['total'] - trends['resolved']
            return trends
        except Exception:
            return pd.DataFrame()
    
    def get_overdue_complaints(self):
        overdue = []
        df = self.get_all_complaints()
        for _, row in df.iterrows():
            if row['status'] in ['Pending', 'In Progress', 'Escalated', 'Assigned'] and row.get('sla_deadline'):
                try:
                    deadline = datetime.strptime(str(row['sla_deadline']), "%Y-%m-%d %H:%M:%S")
                    if datetime.now() > deadline:
                        hours_overdue = (datetime.now() - deadline).total_seconds() / 3600
                        overdue.append({
                            'id': row['id'],
                            'title': row['title'],
                            'severity': row['severity'],
                            'model_severity': row.get('model_severity', 'Medium'),
                            'priority': row.get('priority', 'Medium'),
                            'deadline': row['sla_deadline'],
                            'hours_overdue': round(hours_overdue, 1),
                            'assigned_to': row.get('assigned_to', 'Unassigned')
                        })
                except Exception:
                    pass
        return sorted(overdue, key=lambda x: x['hours_overdue'], reverse=True)
    
    # ------------------------------------------------------------------
    # Sample data
    # ------------------------------------------------------------------
    def _create_sample_data(self):
        """Create sample data with a mix of fresh and OLD complaints to demonstrate escalation"""
        now = datetime.now()
        
        def dt(days_ago, hours=0):
            return (now - timedelta(days=days_ago, hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        
        samples = [
            # Fresh High — no escalation yet
            {
                'id': 'CMP001',
                'name': 'Rahul Sharma',
                'location': 'Mumbai, Maharashtra',
                'datetime': dt(1, 2),
                'title': 'Pothole on Main Road',
                'description': 'Large pothole on Main Road near Andheri station causing traffic jams',
                'primary_category': 'Road & Transportation',
                'subcategory': 'Pothole Repair',
                'model_severity': 'High',
                'severity': 'High',
                'status': 'In Progress',
                'assigned_to': 'BMC Ward Office',
                'department': 'Public Works Department',
                'action_taken': 'Pothole filling in progress',
            },
            # Old Medium (>3 days) → auto-escalates to High
            {
                'id': 'CMP002',
                'name': 'Priya Patel',
                'location': 'Delhi',
                'datetime': dt(4, 0),
                'title': 'Water Leakage in Colony',
                'description': 'Water pipeline burst in Sector 15, water logging on streets',
                'primary_category': 'Water Leakage',
                'subcategory': 'Pipeline Burst',
                'model_severity': 'Medium',
                'severity': 'High',
                'status': 'In Progress',
                'assigned_to': 'Delhi Jal Board',
                'department': 'Water Supply Department',
                'action_taken': 'Pipeline repair ongoing',
            },
            # VERY old High (>3 days) → auto-escalates to Critical
            {
                'id': 'CMP003',
                'name': 'Amit Kumar',
                'location': 'Bengaluru, Karnataka',
                'datetime': dt(5, 6),
                'title': 'Street Light Not Working',
                'description': 'Street lights on MG Road have been non-functional for 2 weeks — dangerous at night',
                'primary_category': 'Electricity',
                'subcategory': 'Street Light Repair',
                'model_severity': 'High',
                'severity': 'Critical',
                'status': 'Pending',
                'assigned_to': 'BBMP',
                'department': 'Electricity Board',
                'action_taken': 'Awaiting inspection',
            },
            # Resolved — should NOT escalate
            {
                'id': 'CMP004',
                'name': 'Sneha Reddy',
                'location': 'Hyderabad, Telangana',
                'datetime': dt(2, 4),
                'title': 'Garbage Collection Issue',
                'description': 'Garbage not collected for 3 days in our locality, foul smell',
                'primary_category': 'Waste Management',
                'subcategory': 'Garbage Collection',
                'model_severity': 'High',
                'severity': 'High',
                'status': 'Resolved',
                'assigned_to': 'GHMC',
                'department': 'Sanitation Department',
                'action_taken': 'Garbage collected',
                'resolution_date': dt(0, 6),
            },
            # Old Low (>7 days) → auto-escalates to Medium
            {
                'id': 'CMP005',
                'name': 'Vikram Singh',
                'location': 'Pune, Maharashtra',
                'datetime': dt(9, 0),
                'title': 'Minor Road Crack',
                'description': 'Small crack on footpath near school gate',
                'primary_category': 'Road & Transportation',
                'subcategory': 'Pothole Repair',
                'model_severity': 'Low',
                'severity': 'Medium',
                'status': 'Pending',
                'assigned_to': 'PMC',
                'department': 'Public Works Department',
                'action_taken': 'Registered',
            },
            # VERY old Low (>14 days) → auto-escalates to High
            {
                'id': 'CMP006',
                'name': 'Anita Joshi',
                'location': 'Chennai, Tamil Nadu',
                'datetime': dt(16, 0),
                'title': 'Overflowing Drain',
                'description': 'Drain near bus stop overflowing, mosquito breeding',
                'primary_category': 'Waste Management',
                'subcategory': 'Garbage Collection',
                'model_severity': 'Low',
                'severity': 'High',
                'status': 'In Progress',
                'assigned_to': 'Chennai Corporation',
                'department': 'Sanitation Department',
                'action_taken': 'Scheduled for cleanup',
            },
        ]
        
        # Populate derived fields
        for s in samples:
            model_sev = s['model_severity']
            computed = compute_current_severity(
                model_sev,
                s['datetime'],
                s['status'],
                None
            )
            s['severity'] = computed['current_severity']
            s['manual_severity'] = None
            s['priority'] = self.calculate_priority(
                s['severity'], s['primary_category'], s['description']
            )
            s['sla_deadline'] = self.calculate_sla_deadline(s['severity'], s['datetime'])
            s['response_time'] = 2.0 if s['status'] != 'Pending' else None
            s['resolution_time'] = None
            s['sla_met'] = None
            s['escalated'] = False
            s['escalation_reason'] = None
            s['feedback_rating'] = 5.0 if s['id'] == 'CMP004' else None
            s['feedback_comments'] = 'Very satisfied' if s['id'] == 'CMP004' else None
            s['schema_version'] = self.SCHEMA_VERSION
            
            # Timeline
            timeline = [{'time': s['datetime'], 'event': 'Complaint registered'}]
            timeline.append({'time': s['datetime'], 'event': f"AI predicted severity: {model_sev}"})
            if computed['escalation_reason']:
                timeline.append({
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'event': f"⬆️ Auto-escalated: {computed['escalation_reason']}"
                })
            timeline.append({
                'time': s['datetime'],
                'event': f"Assigned to {s['assigned_to']}"
            })
            if s.get('resolution_date'):
                timeline.append({
                    'time': s['resolution_date'],
                    'event': f"Resolved - {s['action_taken']}"
                })
            
            # Compute resolution time & sla_met if resolved
            if s['status'] == 'Resolved' and s.get('resolution_date'):
                try:
                    created = datetime.strptime(s['datetime'], "%Y-%m-%d %H:%M:%S")
                    resolved = datetime.strptime(s['resolution_date'], "%Y-%m-%d %H:%M:%S")
                    s['resolution_time'] = round((resolved - created).total_seconds() / 3600, 1)
                    deadline = datetime.strptime(s['sla_deadline'], "%Y-%m-%d %H:%M:%S")
                    s['sla_met'] = resolved <= deadline
                except Exception:
                    pass
            
            s['timeline'] = timeline
        
        return samples


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class MultiTaskModel(nn.Module):
    def __init__(self, base_model, n_primary, n_subcat, n_severity, dropout=0.2):
        super().__init__()
        self.base = base_model
        hidden_size = base_model.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.shared = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.fc_primary = nn.Linear(hidden_size, n_primary)
        self.fc_subcat = nn.Linear(hidden_size, n_subcat)
        self.fc_severity = nn.Linear(hidden_size, n_severity)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.gelu(self.shared(cls_embedding))
        x = self.dropout(x)
        return self.fc_primary(x), self.fc_subcat(x), self.fc_severity(x)


class ModelLoader:
    MODEL_CONFIGS = {
        'HieENBeRT-XLM-RoBERTa': {
            'path': 'hienbert_results/models/best_hienbert_model.pt',
            'tokenizer': 'xlm-roberta-base',
        },
        'IndicBERT': {
            'path': 'indicBert_results/models/best_indicbert_model.pt',
            'tokenizer': 'ai4bharat/indic-bert',
        },
        'MBERT': {
            'path': 'mbert_results/models/best_mbert_model.pt',
            'tokenizer': 'bert-base-multilingual-cased',
        },
        'MURIL': {
            'path': 'muril_results/models/best_muril_model.pt',
            'tokenizer': 'google/muril-base-cased',
        }
    }
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.label_encoders = {}
        self.loaded_models = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_models(self, progress_callback=None):
        for model_name, config in self.MODEL_CONFIGS.items():
            try:
                if progress_callback:
                    progress_callback(f"Checking {model_name}...")
                
                checkpoint_path = os.path.join(os.getcwd(), config['path'])
                if not os.path.exists(checkpoint_path):
                    print(f"Model file not found: {checkpoint_path}")
                    continue
                
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                if not isinstance(checkpoint, dict):
                    continue
                
                n_primary = checkpoint.get('n_primary')
                n_subcat = checkpoint.get('n_subcat')
                n_severity = checkpoint.get('n_severity')
                hidden_size = checkpoint.get('hidden_size')
                
                if n_primary is None and 'model_state_dict' in checkpoint:
                    sd = checkpoint['model_state_dict']
                    for k in sd.keys():
                        if 'fc_primary.weight' in k:
                            n_primary = sd[k].shape[0]
                        elif 'fc_subcat.weight' in k:
                            n_subcat = sd[k].shape[0]
                        elif 'fc_severity.weight' in k:
                            n_severity = sd[k].shape[0]
                        elif 'shared.weight' in k:
                            hidden_size = sd[k].shape[0]
                
                if None in (n_primary, n_subcat, n_severity, hidden_size):
                    continue
                
                if progress_callback:
                    progress_callback(f"Loading {model_name}...")
                
                tokenizer = AutoTokenizer.from_pretrained(
                    config['tokenizer'],
                    token=os.environ.get("HF_TOKEN", None)
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                self.tokenizers[model_name] = tokenizer
                
                base_model = AutoModel.from_pretrained(
                    config['tokenizer'],
                    token=os.environ.get("HF_TOKEN", None)
                ).to(self.device)
                
                model = MultiTaskModel(
                    base_model=base_model,
                    n_primary=n_primary,
                    n_subcat=n_subcat,
                    n_severity=n_severity,
                    dropout=0.2
                ).to(self.device)
                
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                model.eval()
                self.models[model_name] = model
                self.loaded_models.append(model_name)
                print(f"✅ Loaded {model_name}")
                
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                continue
        
        # Label encoders
        if self.loaded_models:
            for model_name in self.loaded_models:
                base_dir = os.path.dirname(self.MODEL_CONFIGS[model_name]['path'])
                for encoder_name in ['le_primary', 'le_subcat', 'le_severity']:
                    if encoder_name in self.label_encoders:
                        continue
                    for path in [
                        os.path.join(base_dir, f'{encoder_name}.pkl'),
                        os.path.join(os.getcwd(), 'hienbert_results', f'{encoder_name}.pkl'),
                        os.path.join(os.getcwd(), 'indicBert_results', f'{encoder_name}.pkl'),
                        os.path.join(os.getcwd(), 'mbert_results', f'{encoder_name}.pkl'),
                        os.path.join(os.getcwd(), 'muril_results', f'{encoder_name}.pkl'),
                    ]:
                        if os.path.exists(path):
                            try:
                                with open(path, 'rb') as f:
                                    self.label_encoders[encoder_name] = pickle.load(f)
                                break
                            except Exception:
                                pass
        
        from sklearn.preprocessing import LabelEncoder
        defaults = {
            'le_primary': ['Road & Transportation', 'Water Leakage', 'Electricity', 'Waste Management', 'Other'],
            'le_subcat': ['Pothole Repair', 'Pipeline Burst', 'Street Light Repair', 'Garbage Collection', 'Other'],
            'le_severity': ['Low', 'Medium', 'High'],
        }
        for enc_name, classes in defaults.items():
            if enc_name not in self.label_encoders:
                le = LabelEncoder()
                le.classes_ = np.array(classes)
                self.label_encoders[enc_name] = le
        
        return self.loaded_models
    
    def predict(self, text):
        if not self.loaded_models:
            return None
        
        results = []
        for model_name in self.loaded_models:
            try:
                tokenizer = self.tokenizers[model_name]
                model = self.models[model_name]
                
                enc = tokenizer(text, truncation=True, padding="max_length",
                                max_length=128, return_tensors="pt")
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                
                with torch.no_grad():
                    p1, p2, p3 = model(input_ids, attention_mask)
                    primary_pred = torch.argmax(p1, dim=1).item()
                    subcat_pred = torch.argmax(p2, dim=1).item()
                    severity_pred = torch.argmax(p3, dim=1).item()
                    primary_prob = torch.softmax(p1, dim=1).max().item()
                
                le_primary = self.label_encoders['le_primary']
                le_subcat = self.label_encoders['le_subcat']
                le_severity = self.label_encoders['le_severity']
                
                primary_idx = min(primary_pred, len(le_primary.classes_) - 1)
                subcat_idx = min(subcat_pred, len(le_subcat.classes_) - 1)
                severity_idx = min(severity_pred, len(le_severity.classes_) - 1)
                
                results.append({
                    'model': model_name,
                    'primary': le_primary.inverse_transform([primary_idx])[0],
                    'subcategory': le_subcat.inverse_transform([subcat_idx])[0],
                    'severity': le_severity.inverse_transform([severity_idx])[0],
                    'confidence': primary_prob,
                })
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                continue
        
        if not results:
            return None
        
        # Internal ensemble voting
        primary_votes, subcat_votes, severity_votes = {}, {}, {}
        conf_sum = 0
        for r in results:
            primary_votes[r['primary']] = primary_votes.get(r['primary'], 0) + 1
            subcat_votes[r['subcategory']] = subcat_votes.get(r['subcategory'], 0) + 1
            severity_votes[r['severity']] = severity_votes.get(r['severity'], 0) + 1
            conf_sum += r['confidence']
        
        final_primary = max(primary_votes, key=primary_votes.get)
        final_subcat = max(subcat_votes, key=subcat_votes.get)
        final_severity = max(severity_votes, key=severity_votes.get)
        
        agreement = max(primary_votes.values()) / len(results)
        
        return {
            'primary': final_primary,
            'subcategory': final_subcat,
            'severity': final_severity,
            'confidence': conf_sum / len(results),
            'agreement': agreement,
            'model_count': len(results),
        }


# ============================================================================
# HELPERS
# ============================================================================
import re

def is_valid_value(val):
    """
    Return True if val is a real, usable value.
    Filters out None, NaN, 'nan', 'none', 'null', ''.
    """
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    s = str(val).strip().lower()
    return s not in ('', 'nan', 'none', 'null')


def safe_int(val, default=0):
    try:
        return int(float(val)) if is_valid_value(val) else default
    except (ValueError, TypeError):
        return default


def render_html(html_str):
    """
    Render HTML safely in Streamlit.
    Collapses newlines/indentation that would otherwise be treated
    as Markdown code blocks.
    """
    collapsed = re.sub(r'\s+', ' ', str(html_str)).strip()
    st.markdown(collapsed, unsafe_allow_html=True)
    
def safe_columns(df, cols):
    return [c for c in cols if c in df.columns]


def render_severity_badge(sev):
    cls = severity_color_class(sev)
    emoji = severity_emoji(sev)
    return f'<span class="sev-badge {cls}">{emoji} {sev}</span>'

def render_html(html_str):
    """Render HTML safely by collapsing whitespace/newlines that break Streamlit markdown."""
    # Collapse all runs of whitespace (including newlines) to a single space
    collapsed = re.sub(r'\s+', ' ', html_str).strip()
    st.markdown(collapsed, unsafe_allow_html=True)
# ============================================================================
# MAIN APP
# ============================================================================

def main():
    load_css()
    
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = ModelLoader()
        st.session_state.models_loaded = False
        st.session_state.complaint_store = ComplaintStore()
        st.session_state.page = 'dashboard'
    
    # Refresh severity escalations on every rerun (cheap)
    st.session_state.complaint_store.refresh_severities()
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h2>📋 CivicRedress-Automated Complaint Management System</h2>
        <p>AI Classification • Dynamic Severity Escalation • SLA Tracking • Officer Override</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("complaint.png", width=120)
        st.markdown("### ⚙️ Navigation")
        
        for label, page in [
            ("📊 Dashboard", "dashboard"),
            ("📝 New Complaint", "new_complaint"),
            ("📋 All Complaints", "view_complaints"),
            ("📈 Analytics", "analytics"),
            ("🚨 Escalations", "escalations"),
        ]:
            if st.button(label, use_container_width=True, key=f"nav_{page}"):
                st.session_state.page = page
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🤖 AI Engine")
        
        if st.button("🚀 Initialize AI", use_container_width=True):
            with st.spinner("Initializing AI engine..."):
                bar = st.progress(0)
                status = st.empty()
                
                def cb(msg):
                    status.text(msg)
                
                loaded = st.session_state.model_loader.load_models(cb)
                st.session_state.models_loaded = len(loaded) > 0
                bar.progress(100)
                
                if loaded:
                    status.text(f"✅ Ready ({len(loaded)} models)")
                    st.success(f"Loaded {len(loaded)} models")
                else:
                    status.text("❌ Failed")
                    st.error("Could not load models.")
                st.rerun()
        
        if st.session_state.models_loaded:
            st.markdown("✅ **AI Engine:** Online")
            st.markdown(f"**Models:** {len(st.session_state.model_loader.loaded_models)}")
        else:
            st.warning("⚠️ AI Engine offline")
        
        st.markdown("---")
        stats = st.session_state.complaint_store.get_statistics()
        st.markdown("### 📈 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", stats['total'])
            st.metric("Resolved", stats['resolved'])
            st.metric("Critical", stats['critical'])
        with col2:
            st.metric("Pending", stats['pending'])
            st.metric("Overdue", stats['pending_overdue'])
            st.metric("Escalated", stats['escalated'])
        
        if stats['auto_escalated'] > 0:
            st.warning(f"⬆️ {stats['auto_escalated']} auto-escalated")
        if stats['manual_overrides'] > 0:
            st.info(f"✏️ {stats['manual_overrides']} officer overrides")
    
    # Routing
    if st.session_state.page == 'dashboard':
        show_dashboard()
    elif st.session_state.page == 'new_complaint':
        show_new_complaint()
    elif st.session_state.page == 'view_complaints':
        show_view_complaints()
    elif st.session_state.page == 'analytics':
        show_analytics()
    elif st.session_state.page == 'escalations':
        show_escalations()


# ============================================================================
# DASHBOARD
# ============================================================================

def show_dashboard():
    st.markdown("### 📊 Dashboard")
    
    stats = st.session_state.complaint_store.get_statistics()
    df = st.session_state.complaint_store.get_all_complaints()
    
    # KPIs - build as single line to avoid markdown code-block issue
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        (c1, stats['total'], "Total", "#2a5298"),
        (c2, stats['resolved'], "✅ Resolved", "#28a745"),
        (c3, stats['pending'], "⏳ Pending", "#ffc107"),
        (c4, stats['pending_overdue'], "🚨 Overdue", "#dc3545"),
        (c5, f"{stats['sla_compliance']:.0f}%", "SLA Met", "#17a2b8"),
        (c6, stats['critical'], "🔴 Critical", "#c62828"),
    ]
    for col, val, label, color in metrics:
        with col:
            html = (
                f'<div class="metric-card" style="border-bottom-color: {color};">'
                f'<div class="metric-value" style="color: {color};">{val}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>'
            )
            render_html(html)
    
    st.markdown("---")
    
    # Charts
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Category")
        if not df.empty and 'primary_category' in df.columns:
            cc = df['primary_category'].value_counts()
            if not cc.empty:
                fig = px.pie(values=cc.values, names=cc.index, hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=280, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### Current Severity")
        if not df.empty and 'severity' in df.columns:
            sc = df['severity'].value_counts()
            colors = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#fd7e14', 'Critical': '#c62828'}
            if not sc.empty:
                fig = px.bar(x=sc.index, y=sc.values, color=sc.index,
                             color_discrete_map=colors)
                fig.update_layout(height=280, showlegend=False,
                                  margin=dict(t=10, b=10),
                                  xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
    with c3:
        st.markdown("#### Status")
        if not df.empty and 'status' in df.columns:
            stc = df['status'].value_counts()
            colors = {'Resolved': '#28a745', 'Pending': '#ffc107',
                      'In Progress': '#17a2b8', 'Escalated': '#dc3545'}
            if not stc.empty:
                fig = px.bar(x=stc.index, y=stc.values, color=stc.index,
                             color_discrete_map=colors)
                fig.update_layout(height=280, showlegend=False,
                                  margin=dict(t=10, b=10),
                                  xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
    
    # Severity drift analysis
    st.markdown("---")
    st.markdown("#### 📊 Severity Drift: Model vs Current")
    
    if not df.empty and 'model_severity' in df.columns and 'severity' in df.columns:
        drift_records = []
        for _, row in df.iterrows():
            if row['status'] != 'Resolved':
                ms = row.get('model_severity')
                cs = row.get('severity')
                man = row.get('manual_severity')
                if ms and cs:
                    drift_records.append({
                        'Model': ms,
                        'Current': cs,
                        'Type': 'Manual Override' if man else ('Auto-Escalated' if ms != cs else 'No Change')
                    })
        
        if drift_records:
            drift_df = pd.DataFrame(drift_records)
            cross = pd.crosstab(drift_df['Model'], drift_df['Current'])
            st.dataframe(cross, use_container_width=True)
        else:
            st.info("No severity drift detected.")
    
    # Recent critical
    st.markdown("---")
    st.markdown("#### 🚨 Critical / Overdue Complaints")
    if not df.empty and 'severity' in df.columns:
        crit = df[(df['severity'] == 'Critical') & (df['status'] != 'Resolved')]
        if not crit.empty:
            for _, row in crit.head(5).iterrows():
                override_text = 'Manual override' if row.get('manual_severity') else 'Auto-escalated'
                html = (
                    f'<div class="critical-alert">'
                    f'🔴 {row["id"]} — {row["title"]} — {row["name"]}'
                    f'<div style="font-size: 0.85rem; margin-top: 0.4rem; font-weight: 500;">'
                    f'Model: {row.get("model_severity")} → Current: <b>CRITICAL</b>'
                    f' | {override_text}'
                    f'</div>'
                    f'</div>'
                )
                render_html(html)
        else:
            st.success("✅ No critical active complaints.")
    
    st.markdown("---")
    st.markdown("#### 📋 Recent Complaints")
    if not df.empty:
        recent = df.sort_values('datetime', ascending=False).head(8).copy()
        
        for _, row in recent.iterrows():
            model_sev = row.get('model_severity', 'Medium')
            cur_sev = row.get('severity', 'Medium')
            is_drift = model_sev != cur_sev and row['status'] != 'Resolved'
            is_manual = bool(row.get('manual_severity'))
            
            drift_badge = ""
            if is_manual:
                drift_badge = f' <span class="overdue-data">✏️ Manual ({model_sev} → {cur_sev})</span>'
            elif is_drift:
                drift_badge = f' <span class="overdue-data">⬆️ Escalated ({model_sev} → {cur_sev})</span>'
            
            row_class = "critical-data" if cur_sev == 'Critical' else ""
            status_class = str(row['status']).lower().replace(' ', '-')
            
            # Build entire HTML as ONE line — no indentation, no newlines
            html = (
                f'<div class="timeline-item" style="padding: 0.8rem 1rem;">'
                f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">'
                f'<div>'
                f'<b class="{row_class}">{row["id"]}</b> — <b>{row["title"]}</b>'
                f'<span style="color:#666;"> • {row["name"]} • {row["datetime"]}</span>'
                f'{drift_badge}'
                f'</div>'
                f'<div style="white-space:nowrap;">'
                f'{render_severity_badge(cur_sev)} '
                f'<span class="status-badge status-{status_class}">{row["status"]}</span>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
            render_html(html)
    
    footer()


# ============================================================================
# NEW COMPLAINT
# ============================================================================

def show_new_complaint():
    st.markdown("### 📝 File New Complaint")
    
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        with st.form("complaint_form", clear_on_submit=True):
            st.markdown("#### 👤 Personal")
            name = st.text_input("Full Name *")
            location = st.text_input("Location *")
            current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.text_input("Date & Time", value=current_dt, disabled=True)
            
            st.markdown("---")
            st.markdown("#### 📄 Complaint")
            title = st.text_input("Title *")
            text = st.text_area("Description *", height=150,
                                placeholder="Describe in English, Hindi, or Hinglish...")
            
            st.caption("💡 Multilingual support enabled")
            submitted = st.form_submit_button("🔍 Analyze & Submit")
    
    with c2:
        if submitted:
            if not name.strip() or not location.strip() or not text.strip():
                st.warning("⚠️ Please fill all required fields.")
            elif not st.session_state.models_loaded:
                st.warning("⚠️ Initialize AI engine first.")
            else:
                with st.spinner("Analyzing..."):
                    full_text = f"{title} {text}".strip()
                    result = st.session_state.model_loader.predict(full_text)
                    if result:
                        cid = st.session_state.complaint_store.add_complaint({
                            'name': name.strip(),
                            'location': location.strip(),
                            'datetime': current_dt,
                            'title': title.strip(),
                            'description': text.strip(),
                            'primary_category': result['primary'],
                            'subcategory': result['subcategory'],
                            'severity': result['severity'],
                        })
                        st.session_state.complaint_store.refresh_severities()
                        df = st.session_state.complaint_store.get_all_complaints()
                        saved = df[df['id'] == cid].iloc[0].to_dict() if not df[df['id'] == cid].empty else {}
                        show_prediction_results(result, cid, saved)
                    else:
                        st.error("❌ Prediction failed.")
        else:
            st.markdown("### 📊 AI Prediction")
            st.info("Fill the form and submit to see AI predictions here.")


def show_prediction_results(result, cid, data):
    st.success(f"✅ Complaint **{cid}** registered")
    
    model_sev = data.get('model_severity', result['severity'])
    cur_sev = data.get('severity', result['severity'])
    
    st.markdown("#### 🎯 Severity Analysis")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"""
        <div class="severity-box severity-model">
            <div class="label">🤖 Model Predicted</div>
            <div class="value">{severity_emoji(model_sev)} {model_sev}</div>
            <div class="note">Original AI prediction (kept for retraining)</div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
        <div class="severity-box severity-current">
            <div class="label">📌 Current Severity</div>
            <div class="value">{severity_emoji(cur_sev)} {cur_sev}</div>
            <div class="note">Effective severity right now</div>
        </div>
        """, unsafe_allow_html=True)
    
    if model_sev != cur_sev:
        st.info(f"⬆️ Escalation applied: {model_sev} → {cur_sev}")
    else:
        st.success("✅ No escalation — severity matches AI prediction.")
    
    st.markdown("---")
    st.markdown("#### 📋 Classification Results")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Primary Category", result['primary'], f"{result['confidence']:.1%}")
    with c2:
        st.metric("Subcategory", result['subcategory'])
    with c3:
        st.metric("Model Agreement", f"{result['agreement']:.0%}",
                  f"{result['model_count']} models")
    
    st.markdown("---")
    st.markdown("#### 🤖 Automated Workflow")
    st.markdown(f"- **Department:** {data.get('department', 'N/A')}")
    st.markdown(f"- **Assigned To:** {data.get('assigned_to', 'N/A')}")
    st.markdown(f"- **Priority:** {data.get('priority', 'Medium')}")
    st.markdown(f"- **SLA Deadline:** {data.get('sla_deadline', 'N/A')}")
    
    st.markdown("---")
    st.markdown("#### 📅 Timeline")
    for item in data.get('timeline', []):
        st.markdown(f"""
        <div class="timeline-item">
            <div class="time">{item.get('time', '')}</div>
            <div class="event">{item.get('event', '')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📋 View All Complaints", use_container_width=True):
            st.session_state.page = 'view_complaints'
            st.rerun()
    with c2:
        if st.button("📝 File Another", use_container_width=True):
            st.session_state.page = 'new_complaint'
            st.rerun()


# ============================================================================
# VIEW COMPLAINTS
# ============================================================================

def show_view_complaints():
    st.markdown("### 📋 All Complaints")
    
    stats = st.session_state.complaint_store.get_statistics()
    
    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        status_filter = st.selectbox(
            "Status",
            ['All', 'Pending', 'In Progress', 'Resolved', 'Escalated'],
            key="vc_status_filter"
        )
    with c2:
        severity_filter = st.selectbox(
            "Current Severity",
            ['All', 'Critical', 'High', 'Medium', 'Low'],
            key="vc_severity_filter"
        )
    with c3:
        model_severity_filter = st.selectbox(
            "Model Severity",
            ['All', 'High', 'Medium', 'Low'],
            key="vc_model_severity_filter"
        )
    with c4:
        priority_filter = st.selectbox(
            "Priority",
            ['All', 'Critical', 'High', 'Medium', 'Low'],
            key="vc_priority_filter"
        )
    with c5:
        category_filter = st.selectbox(
            "Category",
            ['All'] + list(stats['categories'].keys()),
            key="vc_category_filter"
        )
    
    search = st.text_input(
        "🔍 Search",
        placeholder="ID, name, title, description...",
        key="vc_search"
    )
    
    c1, c2 = st.columns([1, 3])
    with c1:
        only_escalated = st.checkbox("Only auto-escalated", key="vc_only_esc")
    with c2:
        only_manual = st.checkbox("Only manual overrides", key="vc_only_man")
    
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = st.session_state.complaint_store.get_all_complaints()
    if df.empty:
        st.info("No complaints.")
        return
    
    # ------------------------------------------------------------------
    # Apply filters (all NaN-safe)
    # ------------------------------------------------------------------
    if status_filter != 'All' and 'status' in df.columns:
        df = df[df['status'] == status_filter]
    
    if severity_filter != 'All' and 'severity' in df.columns:
        df = df[df['severity'] == severity_filter]
    
    if model_severity_filter != 'All' and 'model_severity' in df.columns:
        df = df[df['model_severity'] == model_severity_filter]
    
    if priority_filter != 'All' and 'priority' in df.columns:
        df = df[df['priority'] == priority_filter]
    
    if category_filter != 'All' and 'primary_category' in df.columns:
        df = df[df['primary_category'] == category_filter]
    
    # NaN-safe "only escalated" filter
    if only_escalated:
        if 'model_severity' in df.columns and 'severity' in df.columns:
            mask = df['model_severity'].astype(str) != df['severity'].astype(str)
            if 'manual_severity' in df.columns:
                mask = mask & df['manual_severity'].isna()
            df = df[mask]
    
    # NaN-safe "only manual" filter
    if only_manual:
        if 'manual_severity' in df.columns:
            df = df[df['manual_severity'].notna()]
    
    # NaN-safe search
    if search:
        s = search.lower()
        mask = (
            df['id'].astype(str).str.lower().str.contains(s, na=False) |
            df['name'].astype(str).str.lower().str.contains(s, na=False) |
            df['title'].astype(str).str.lower().str.contains(s, na=False) |
            df['description'].astype(str).str.lower().str.contains(s, na=False)
        )
        df = df[mask]
    
    st.markdown(f"**Showing {len(df)} complaints**")
    
    if df.empty:
        st.info("No complaints match filters.")
        return
    
    # ------------------------------------------------------------------
    # Render each complaint
    # ------------------------------------------------------------------
    for _, row in df.iterrows():
        # --- Overdue check ---
        is_overdue = False
        sla_deadline = row.get('sla_deadline')
        if (row['status'] in ['Pending', 'In Progress', 'Escalated', 'Assigned']
                and is_valid_value(sla_deadline)):
            try:
                deadline = datetime.strptime(str(sla_deadline), "%Y-%m-%d %H:%M:%S")
                is_overdue = datetime.now() > deadline
            except Exception:
                pass
        
        # --- Severity (all NaN-safe) ---
        model_sev = row.get('model_severity')
        if not is_valid_value(model_sev):
            model_sev = 'Medium'
        model_sev = str(model_sev).title()
        
        cur_sev = row.get('severity')
        if not is_valid_value(cur_sev):
            cur_sev = model_sev
        cur_sev = str(cur_sev).title()
        
        manual_sev = row.get('manual_severity')
        has_manual = is_valid_value(manual_sev)
        if has_manual:
            manual_sev = str(manual_sev).title()
        
        # --- Drift marker ---
        drift_marker = ""
        if has_manual:
            drift_marker = f" ✏️ Override ({model_sev}→{manual_sev})"
        elif model_sev != cur_sev and row['status'] != 'Resolved':
            drift_marker = f" ⬆️ Auto ({model_sev}→{cur_sev})"
        
        overdue_marker = " 🚨 OVERDUE" if is_overdue else ""
        
        expander_label = (
            f"{row['id']} - {row['title']} [{row['status']}]"
            f"{drift_marker}{overdue_marker}"
        )
        
        with st.expander(expander_label):
            left, right = st.columns([2, 1])
            
            # =========================================================
            # LEFT COLUMN
            # =========================================================
            with left:
                st.markdown(f"**👤 {row['name']}** • 📍 {row['location']}")
                st.markdown(f"**📅 Registered:** {row['datetime']}")
                
                dept = row.get('department')
                dept = dept if is_valid_value(dept) else 'N/A'
                st.markdown(f"**🏢 Department:** {dept}")
                
                assigned = row.get('assigned_to')
                assigned = assigned if is_valid_value(assigned) else 'Unassigned'
                st.markdown(f"**👥 Assigned To:** {assigned}")
                
                st.markdown("---")
                
                # --- Dual Severity Display ---
                st.markdown("##### 🎯 Severity Analysis")
                sc1, sc2 = st.columns(2)
                with sc1:
                    html = (
                        f'<div class="severity-box severity-model" style="padding: 0.8rem;">'
                        f'<div class="label">🤖 Model</div>'
                        f'<div class="value" style="font-size: 1.4rem;">'
                        f'{severity_emoji(model_sev)} {model_sev}'
                        f'</div>'
                        f'</div>'
                    )
                    render_html(html)
                with sc2:
                    box_class = "severity-critical" if cur_sev == 'Critical' else "severity-current"
                    html = (
                        f'<div class="severity-box {box_class}" style="padding: 0.8rem;">'
                        f'<div class="label">📌 Current</div>'
                        f'<div class="value" style="font-size: 1.4rem;">'
                        f'{severity_emoji(cur_sev)} {cur_sev}'
                        f'</div>'
                        f'</div>'
                    )
                    render_html(html)
                
                # --- Escalation / override notice ---
                if has_manual:
                    st.warning(f"✏️ **Officer override active:** {model_sev} → {manual_sev}")
                elif model_sev != cur_sev and row['status'] != 'Resolved':
                    days = 0
                    try:
                        d = datetime.strptime(str(row['datetime']), "%Y-%m-%d %H:%M:%S")
                        days = round((datetime.now() - d).total_seconds() / 86400, 1)
                    except Exception:
                        pass
                    st.warning(f"⬆️ **Auto-escalated** after {days} days: {model_sev} → {cur_sev}")
                
                st.markdown("---")
                st.markdown(f"**📌 Category:** {row['primary_category']} → {row['subcategory']}")
                
                priority = row.get('priority')
                priority = str(priority) if is_valid_value(priority) else 'Medium'
                st.markdown(
                    f'**🎯 Priority:** <span class="priority-{priority.lower()}">{priority}</span>',
                    unsafe_allow_html=True
                )
                
                st.markdown("---")
                st.markdown("**📝 Description:**")
                st.markdown(str(row['description']) if is_valid_value(row.get('description')) else "_No description_")
                
                action = row.get('action_taken')
                action = action if is_valid_value(action) else 'N/A'
                st.markdown(f"**🔧 Action:** {action}")
                
                # --- SLA ---
                if is_valid_value(sla_deadline):
                    sla_met_val = row.get('sla_met')
                    if sla_met_val is True:
                        sla_status = "✅ Met"
                    elif sla_met_val is False:
                        sla_status = "❌ Missed"
                    else:
                        sla_status = "⏳ Pending"
                    sla_class = "overdue-data" if is_overdue else ""
                    st.markdown(
                        f'**⏱️ SLA Deadline:** '
                        f'<span class="{sla_class}">{sla_deadline}</span> ({sla_status})',
                        unsafe_allow_html=True
                    )
                
                # --- Resolution ---
                res_date = row.get('resolution_date')
                if is_valid_value(res_date):
                    st.markdown(f"**✅ Resolved:** {res_date}")
                
                # --- Feedback (NaN-safe) ---
                fb_rating = row.get('feedback_rating')
                if is_valid_value(fb_rating):
                    try:
                        rating_int = int(float(fb_rating))
                        stars = "⭐" * max(0, min(rating_int, 5))
                        st.markdown(f"**⭐ Feedback:** {stars} ({rating_int}/5)")
                        fb_comments = row.get('feedback_comments')
                        if is_valid_value(fb_comments):
                            st.markdown(f'*"{fb_comments}"*')
                    except (ValueError, TypeError):
                        pass
            
            # =========================================================
            # RIGHT COLUMN
            # =========================================================
            with right:
                # --- Severity Override ---
                st.markdown("#### ✏️ Severity Override")
                with st.form(f"sev_override_{row['id']}"):
                    st.caption("Preserves AI prediction for retraining")
                    sev_options = ['Low', 'Medium', 'High', 'Critical']
                    try:
                        default_idx = sev_options.index(cur_sev)
                    except ValueError:
                        default_idx = 1
                    new_sev = st.selectbox(
                        "Assign Severity",
                        sev_options,
                        index=default_idx,
                        key=f"sevsel_{row['id']}"
                    )
                    override_reason = st.text_input(
                        "Reason",
                        key=f"sevreason_{row['id']}"
                    )
                    
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        if st.form_submit_button("✅ Apply", key=f"sevapply_{row['id']}"):
                            st.session_state.complaint_store.override_severity(
                                row['id'], new_sev,
                                override_reason if override_reason.strip() else None
                            )
                            st.success(f"Severity overridden to {new_sev}")
                            st.rerun()
                    with cc2:
                        if has_manual:
                            if st.form_submit_button("↩️ Clear", key=f"sevclear_{row['id']}"):
                                st.session_state.complaint_store.clear_severity_override(row['id'])
                                st.success("Override cleared")
                                st.rerun()
                
                # --- Status Update ---
                st.markdown("#### 🔄 Update Status")
                with st.form(f"status_form_{row['id']}"):
                    statuses = ['Pending', 'In Progress', 'Resolved', 'Escalated']
                    try:
                        st_idx = statuses.index(row['status'])
                    except ValueError:
                        st_idx = 0
                    new_status = st.selectbox(
                        "Status",
                        statuses,
                        index=st_idx,
                        key=f"st_{row['id']}"
                    )
                    
                    current_action = row.get('action_taken')
                    action_default = (
                        current_action
                        if is_valid_value(current_action)
                        and current_action not in ('Awaiting action', None)
                        else ''
                    )
                    action = st.text_input(
                        "Action Taken",
                        value=action_default,
                        key=f"act_{row['id']}"
                    )
                    
                    current_assigned = row.get('assigned_to')
                    assigned_default = (
                        current_assigned
                        if is_valid_value(current_assigned)
                        and current_assigned not in ('Not assigned', None)
                        else ''
                    )
                    assigned = st.text_input(
                        "Assigned To",
                        value=assigned_default,
                        key=f"asgn_{row['id']}"
                    )
                    
                    if st.form_submit_button("✅ Update", key=f"upd_{row['id']}"):
                        st.session_state.complaint_store.update_status(
                            row['id'], new_status,
                            action if action.strip() else None,
                            assigned if assigned.strip() else None
                        )
                        st.success("Updated!")
                        st.rerun()
                
                # --- Escalation ---
                if row['status'] not in ['Resolved', 'Escalated']:
                    st.markdown("#### 🚨 Escalate")
                    with st.form(f"esc_{row['id']}"):
                        reason = st.text_area(
                            "Reason",
                            key=f"escr_{row['id']}",
                            height=70
                        )
                        if st.form_submit_button("🚨 Escalate", key=f"escb_{row['id']}"):
                            if reason.strip():
                                st.session_state.complaint_store.escalate_complaint(
                                    row['id'], reason.strip()
                                )
                                st.success("Escalated")
                                st.rerun()
                            else:
                                st.warning("Provide a reason.")
                
                # --- Feedback form (only if resolved AND no feedback) ---
                fb_rating = row.get('feedback_rating')
                has_feedback = is_valid_value(fb_rating)
                if row['status'] == 'Resolved' and not has_feedback:
                    st.markdown("#### ⭐ Feedback")
                    with st.form(f"fb_{row['id']}"):
                        rating = st.slider("Rating", 1, 5, 3, key=f"r_{row['id']}")
                        comments = st.text_area(
                            "Comments",
                            key=f"c_{row['id']}",
                            height=60
                        )
                        if st.form_submit_button("Submit", key=f"fbb_{row['id']}"):
                            st.session_state.complaint_store.add_feedback(
                                row['id'], rating,
                                comments if comments.strip() else None
                            )
                            st.success("Thanks!")
                            st.rerun()
                
                # --- Timeline (single-line HTML to avoid markdown code blocks) ---
                st.markdown("#### 📅 Timeline")
                timeline = row.get('timeline', [])
                if isinstance(timeline, list) and timeline:
                    for item in timeline[-6:]:
                        t_time = item.get('time', '') if isinstance(item, dict) else ''
                        t_event = item.get('event', '') if isinstance(item, dict) else ''
                        html = (
                            f'<div class="timeline-item" style="font-size: 0.82rem;">'
                            f'<div class="time">{t_time}</div>'
                            f'<div class="event">{t_event}</div>'
                            f'</div>'
                        )
                        render_html(html)

# ============================================================================
# ANALYTICS
# ============================================================================

def show_analytics():
    st.markdown("### 📈 Analytics")
    
    stats = st.session_state.complaint_store.get_statistics()
    df = st.session_state.complaint_store.get_all_complaints()
    if df.empty:
        st.info("No data.")
        return
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Avg Response", f"{stats['avg_response_time']:.1f} hrs")
    with c2:
        st.metric("Avg Resolution", f"{stats['avg_resolution_time']:.1f} hrs")
    with c3:
        st.metric("SLA Compliance", f"{stats['sla_compliance']:.1f}%")
    with c4:
        st.metric("Avg Rating", f"{stats['avg_rating']:.1f}/5.0" if stats['avg_rating'] else "N/A")
    
    st.markdown("---")
    st.markdown("#### 🔄 Severity Escalation Impact")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-bottom-color: #1976d2;">
            <div class="metric-value" style="color: #1976d2;">{stats['total']}</div>
            <div class="metric-label">Total Complaints</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-bottom-color: #f57c00;">
            <div class="metric-value" style="color: #f57c00;">{stats['auto_escalated']}</div>
            <div class="metric-label">⬆️ Auto-Escalated</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card" style="border-bottom-color: #6f42c1;">
            <div class="metric-value" style="color: #6f42c1;">{stats['manual_overrides']}</div>
            <div class="metric-label">✏️ Manual Overrides</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 📊 Model vs Current Severity Matrix")
    if 'model_severity' in df.columns and 'severity' in df.columns:
        crosstab = pd.crosstab(df['model_severity'], df['severity'],
                               rownames=['Model'], colnames=['Current'])
        st.dataframe(crosstab, use_container_width=True)
        st.caption("Diagonal = agreement. Off-diagonal = escalations or overrides.")
    
    st.markdown("---")
    st.markdown("#### 🏢 Department Performance")
    if 'department' in df.columns:
        dept = df.groupby('department').agg(
            Total=('id', 'count'),
            Resolved=('status', lambda x: (x == 'Resolved').sum()),
        )
        dept['Resolution %'] = (dept['Resolved'] / dept['Total'] * 100).round(1)
        st.dataframe(dept, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 📈 30-Day Trend")
    trends = st.session_state.complaint_store.get_trends(30)
    if not trends.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=trends.index, y=trends['total'], name='New',
                             marker_color='#2a5298'))
        fig.add_trace(go.Bar(x=trends.index, y=trends['resolved'], name='Resolved',
                             marker_color='#28a745'))
        fig.update_layout(height=350, barmode='group', margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    footer()


# ============================================================================
# ESCALATIONS
# ============================================================================

def show_escalations():
    st.markdown("### 🚨 Escalation Management")
    
    overdue = st.session_state.complaint_store.get_overdue_complaints()
    
    if overdue:
        st.markdown(f"""
        <div class="critical-alert">
            🚨 {len(overdue)} COMPLAINT(S) OVERDUE — IMMEDIATE ATTENTION REQUIRED
        </div>
        """, unsafe_allow_html=True)
        
        for item in overdue:
            cur = item['severity']
            model = item['model_severity']
            drift = f" (Model: {model} → Current: {cur})" if model != cur else ""
            
            st.markdown(f"""
            <div class="card" style="border-left-color: #c62828; background: #fff5f5;">
                <div style="display:flex; justify-content:space-between; flex-wrap:wrap;">
                    <div>
                        <h4 class="critical-data">{item['id']} — {item['title']}</h4>
                        <p><b>Assigned To:</b> {item['assigned_to']}</p>
                        <p><b>Deadline:</b> {item['deadline']}</p>
                        <p class="critical-data"><b>OVERDUE BY: {item['hours_overdue']} HOURS</b></p>
                        <p style="color:#c62828; font-weight:600;">Severity: {cur}{drift}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ No overdue complaints.")
    
    st.markdown("---")
    df = st.session_state.complaint_store.get_all_complaints()
    if not df.empty and 'escalated' in df.columns:
        esc = df[df['escalated'].fillna(False).astype(bool)]
        if not esc.empty:
            st.markdown(f"#### 🚨 Manually Escalated ({len(esc)})")
            for _, row in esc.iterrows():
                with st.expander(f"🚨 {row['id']} - {row['title']}"):
                    st.markdown(f"**Name:** {row['name']}")
                    st.markdown(f"**Reason:** {row.get('escalation_reason', 'N/A')}")
                    st.markdown(f"**Status:** {row['status']}")
        else:
            st.info("No manually escalated complaints.")
    
    footer()


def footer():
    st.markdown("""
    <div class="footer">
        <p class="creator">👨‍💻 Created by: Shyam Tripathi (CH.SC.U4CSE23160) & Kothamasu Hridyaesh (CH.SC.U4CSE23117)</p>
        <p class="guide">📚 Guide: Dr. K. Ashwini (Assistant Professor Senior Grade)</p>
        <p style="color: #888; font-size: 0.9rem; margin-top: 0.5rem;">
            © 2026 Automated Complaint Management System | Dynamic Severity Escalation
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()