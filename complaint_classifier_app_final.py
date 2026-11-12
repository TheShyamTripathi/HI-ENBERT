# complaint_classifier_app.py - Fixed Version

import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import os
from datetime import datetime
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
    page_title="Complaint Classification System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR STYLING
# ============================================================================

def load_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .header-container h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .header-container p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #2a5298;
    }
    .card h3 {
        color: #1e3c72;
        margin-bottom: 0.5rem;
    }
    
    /* Result card styling */
    .result-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 5px solid #28a745;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .result-card h2 {
        color: #1e3c72;
        margin-bottom: 1rem;
    }
    .result-card .prediction {
        font-size: 1.2rem;
        padding: 0.5rem;
        background: white;
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    .result-card .prediction strong {
        color: #2a5298;
    }
    
    /* Model badge */
    .model-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .model-badge.xlmr { background: #e3f2fd; color: #0d47a1; }
    .model-badge.indic { background: #e8f5e9; color: #1b5e20; }
    .model-badge.mbert { background: #fff3e0; color: #e65100; }
    .model-badge.muril { background: #fce4ec; color: #880e4f; }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-badge.resolved { background: #d4edda; color: #155724; }
    .status-badge.pending { background: #fff3cd; color: #856404; }
    .status-badge.in-progress { background: #cce5ff; color: #004085; }
    .status-badge.rejected { background: #f8d7da; color: #721c24; }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 2rem;
        background: #f8f9fa;
        border-radius: 12px;
        border-top: 3px solid #2a5298;
    }
    .footer .creator {
        font-weight: 600;
        color: #1e3c72;
    }
    .footer .guide {
        color: #555;
        font-style: italic;
    }
    
    /* Sidebar styling */
    .sidebar-info {
        padding: 1rem 0;
        border-bottom: 1px solid #dee2e6;
    }
    .sidebar-info:last-child {
        border-bottom: none;
    }
    
    /* Custom button */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border-radius: 25px;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(42, 82, 152, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
        border-bottom: 3px solid #2a5298;
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3c72;
    }
    .metric-card .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATA STORE FOR COMPLAINTS
# ============================================================================

class ComplaintStore:
    """Store and manage complaints with action tracking"""
    
    def __init__(self, storage_file='complaints_data.json'):
        self.storage_file = storage_file
        self.complaints = []
        self.load_data()
    
    def load_data(self):
        """Load complaints from JSON file"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    self.complaints = json.load(f)
            else:
                # Create with sample data
                self.complaints = [
                    {
                        'id': 'CMP001',
                        'name': 'Rahul Sharma',
                        'location': 'Mumbai, Maharashtra',
                        'datetime': '2026-08-20 10:30:00',
                        'title': 'Pothole on Main Road',
                        'description': 'Large pothole on Main Road near Andheri station causing traffic jams',
                        'primary_category': 'Road & Transportation',
                        'subcategory': 'Pothole Repair',
                        'severity': 'High',
                        'action_taken': 'Pothole filled',
                        'status': 'Resolved',
                        'resolution_date': '2026-08-22 14:00:00',
                        'assigned_to': 'BMC Ward Office'
                    },
                    {
                        'id': 'CMP002',
                        'name': 'Priya Patel',
                        'location': 'Delhi',
                        'datetime': '2026-08-21 09:15:00',
                        'title': 'Water Leakage in Colony',
                        'description': 'Water pipeline burst in Sector 15, water logging on streets',
                        'primary_category': 'Water Leakage',
                        'subcategory': 'Pipeline Burst',
                        'severity': 'High',
                        'action_taken': 'Pipeline repair in progress',
                        'status': 'In Progress',
                        'resolution_date': None,
                        'assigned_to': 'Delhi Jal Board'
                    },
                    {
                        'id': 'CMP003',
                        'name': 'Amit Kumar',
                        'location': 'Bengaluru, Karnataka',
                        'datetime': '2026-08-21 11:45:00',
                        'title': 'Street Light Not Working',
                        'description': 'Street lights on MG Road have been non-functional for 2 weeks',
                        'primary_category': 'Electricity',
                        'subcategory': 'Street Light Repair',
                        'severity': 'Medium',
                        'action_taken': 'Complaint registered with BBMP',
                        'status': 'Pending',
                        'resolution_date': None,
                        'assigned_to': 'BBMP'
                    },
                    {
                        'id': 'CMP004',
                        'name': 'Sneha Reddy',
                        'location': 'Hyderabad, Telangana',
                        'datetime': '2026-08-22 08:00:00',
                        'title': 'Garbage Collection Issue',
                        'description': 'Garbage not collected for 3 days in our locality, foul smell',
                        'primary_category': 'Waste Management',
                        'subcategory': 'Garbage Collection',
                        'severity': 'High',
                        'action_taken': 'GHMC vehicles dispatched',
                        'status': 'Resolved',
                        'resolution_date': '2026-08-23 16:30:00',
                        'assigned_to': 'GHMC'
                    }
                ]
                self.save_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            self.complaints = []
    
    def save_data(self):
        """Save complaints to JSON file"""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.complaints, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def add_complaint(self, complaint_data):
        """Add new complaint"""
        # Generate ID
        max_id = 0
        for c in self.complaints:
            try:
                num = int(c['id'].replace('CMP', ''))
                if num > max_id:
                    max_id = num
            except:
                pass
        complaint_data['id'] = f"CMP{max_id + 1:03d}"
        complaint_data['status'] = 'Pending'
        complaint_data['action_taken'] = 'Awaiting action'
        complaint_data['resolution_date'] = None
        complaint_data['assigned_to'] = 'Not assigned'
        
        self.complaints.append(complaint_data)
        self.save_data()
        return complaint_data['id']
    
    def update_status(self, complaint_id, status, action_taken=None, assigned_to=None):
        """Update complaint status"""
        for c in self.complaints:
            if c['id'] == complaint_id:
                c['status'] = status
                if action_taken:
                    c['action_taken'] = action_taken
                if assigned_to:
                    c['assigned_to'] = assigned_to
                if status == 'Resolved':
                    c['resolution_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.save_data()
                return True
        return False
    
    def get_all_complaints(self):
        """Get all complaints as DataFrame"""
        return pd.DataFrame(self.complaints)
    
    def get_statistics(self):
        """Get complaint statistics"""
        df = self.get_all_complaints()
        if df.empty:
            return {
                'total': 0,
                'resolved': 0,
                'pending': 0,
                'in_progress': 0,
                'categories': {}
            }
        return {
            'total': len(df),
            'resolved': len(df[df['status'] == 'Resolved']),
            'pending': len(df[df['status'] == 'Pending']),
            'in_progress': len(df[df['status'] == 'In Progress']),
            'categories': df['primary_category'].value_counts().to_dict()
        }


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class MultiTaskModel(nn.Module):
    """Base model for multi-task classification"""
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
        
        self.n_primary = n_primary
        self.n_subcat = n_subcat
        self.n_severity = n_severity
        self.hidden_size = hidden_size
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        x = self.gelu(self.shared(cls_embedding))
        x = self.dropout(x)
        return self.fc_primary(x), self.fc_subcat(x), self.fc_severity(x)


class ModelLoader:
    """Load and manage multiple models"""
    
    MODEL_CONFIGS = {
        'HieENBeRT-XLM-RoBERTa': {
            'path': 'hienbert_results/models/best_hienbert_model.pt',
            'tokenizer': 'xlm-roberta-base',
            'color': '#0d47a1',
            'badge': 'xlmr',
            'model_type': 'xlmr'
        },
        'IndicBERT': {
            'path': 'indicBert_results/models/best_indicbert_model.pt',
            'tokenizer': 'ai4bharat/indic-bert',
            'color': '#1b5e20',
            'badge': 'indic',
            'model_type': 'indic'
        },
        'MBERT': {
            'path': 'mbert_results/models/best_mbert_model.pt',
            'tokenizer': 'bert-base-multilingual-cased',
            'color': '#e65100',
            'badge': 'mbert',
            'model_type': 'mbert'
        },
        'MURIL': {
            'path': 'muril_results/models/best_muril_model.pt',
            'tokenizer': 'google/muril-base-cased',
            'color': '#880e4f',
            'badge': 'muril',
            'model_type': 'muril'
        }
    }
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.label_encoders = {}
        self.model_metadata = {}
        self.loaded_models = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_paths_checked = {}
        
    def load_models(self, progress_callback=None):
        """Load all available models"""
        for model_name, config in self.MODEL_CONFIGS.items():
            try:
                if progress_callback:
                    progress_callback(f"Checking {model_name}...")
                
                checkpoint_path = os.path.join(os.getcwd(), config['path'])
                if not os.path.exists(checkpoint_path):
                    print(f"Model file not found: {checkpoint_path}")
                    continue
                
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                
                # Check if checkpoint has required keys
                if not isinstance(checkpoint, dict):
                    print(f"Invalid checkpoint format for {model_name}")
                    continue
                
                # Try to get model dimensions from checkpoint or infer from model state dict
                n_primary = checkpoint.get('n_primary', None)
                n_subcat = checkpoint.get('n_subcat', None)
                n_severity = checkpoint.get('n_severity', None)
                hidden_size = checkpoint.get('hidden_size', None)
                
                # If not found in checkpoint, try to infer from label encoders
                if n_primary is None or n_subcat is None or n_severity is None:
                    if 'label_encoders' in checkpoint:
                        label_encoders = checkpoint['label_encoders']
                        if isinstance(label_encoders, dict):
                            n_primary = len(label_encoders.get('le_primary', {}).get('classes_', [])) if 'le_primary' in label_encoders else None
                            n_subcat = len(label_encoders.get('le_subcat', {}).get('classes_', [])) if 'le_subcat' in label_encoders else None
                            n_severity = len(label_encoders.get('le_severity', {}).get('classes_', [])) if 'le_severity' in label_encoders else None
                
                # If still None, try to infer from model state dict
                if n_primary is None and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    for key in state_dict.keys():
                        if 'fc_primary.weight' in key:
                            n_primary = state_dict[key].shape[0]
                        elif 'fc_subcat.weight' in key:
                            n_subcat = state_dict[key].shape[0]
                        elif 'fc_severity.weight' in key:
                            n_severity = state_dict[key].shape[0]
                
                # If hidden_size is None, try to infer
                if hidden_size is None and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    for key in state_dict.keys():
                        if 'shared.weight' in key:
                            hidden_size = state_dict[key].shape[0]
                            break
                
                # Final validation
                if n_primary is None or n_subcat is None or n_severity is None or hidden_size is None:
                    print(f"Cannot determine model dimensions for {model_name}")
                    print(f"n_primary: {n_primary}, n_subcat: {n_subcat}, n_severity: {n_severity}, hidden_size: {hidden_size}")
                    continue
                
                if progress_callback:
                    progress_callback(f"Loading {model_name}...")
                
                # Load tokenizer
                try:
                    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'], token=os.environ.get("HF_TOKEN", None))
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    self.tokenizers[model_name] = tokenizer
                except Exception as e:
                    print(f"Error loading tokenizer for {model_name}: {e}")
                    continue
                
                # Load base model
                try:
                    base_model = AutoModel.from_pretrained(config['tokenizer'], token=os.environ.get("HF_TOKEN", None)).to(self.device)
                except Exception as e:
                    print(f"Error loading base model for {model_name}: {e}")
                    continue
                
                # Create model
                model = MultiTaskModel(
                    base_model=base_model,
                    n_primary=n_primary,
                    n_subcat=n_subcat,
                    n_severity=n_severity,
                    dropout=0.2
                ).to(self.device)
                
                # Load state dict
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                model.eval()
                self.models[model_name] = model
                
                self.model_metadata[model_name] = {
                    'hidden_size': hidden_size,
                    'n_primary': n_primary,
                    'n_subcat': n_subcat,
                    'n_severity': n_severity
                }
                
                self.loaded_models.append(model_name)
                print(f"✅ Successfully loaded {model_name}")
                
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Load label encoders - try multiple paths
        if self.loaded_models:
            for model_name in self.loaded_models:
                model_path = self.MODEL_CONFIGS[model_name]['path']
                base_dir = os.path.dirname(model_path)
                
                for encoder_name in ['le_primary', 'le_subcat', 'le_severity']:
                    if encoder_name in self.label_encoders:
                        continue
                    try:
                        # Try different paths
                        possible_paths = [
                            os.path.join(base_dir, f'{encoder_name}.pkl'),
                            os.path.join(os.getcwd(), 'hienbert_results', f'{encoder_name}.pkl'),
                            os.path.join(os.getcwd(), 'indicBert_results', f'{encoder_name}.pkl'),
                            os.path.join(os.getcwd(), 'mbert_results', f'{encoder_name}.pkl'),
                            os.path.join(os.getcwd(), 'muril_results', f'{encoder_name}.pkl'),
                        ]
                        
                        for path in possible_paths:
                            if os.path.exists(path):
                                with open(path, 'rb') as f:
                                    self.label_encoders[encoder_name] = pickle.load(f)
                                    print(f"✅ Loaded {encoder_name} from {path}")
                                    break
                    except Exception as e:
                        print(f"Error loading {encoder_name}: {e}")
                        continue
                
                # Also try to get label encoders from checkpoint
                try:
                    checkpoint_path = os.path.join(os.getcwd(), model_path)
                    if os.path.exists(checkpoint_path):
                        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                        if 'label_encoders' in checkpoint:
                            for enc_name in ['le_primary', 'le_subcat', 'le_severity']:
                                if enc_name not in self.label_encoders and enc_name in checkpoint['label_encoders']:
                                    self.label_encoders[enc_name] = checkpoint['label_encoders'][enc_name]
                                    print(f"✅ Loaded {enc_name} from checkpoint")
                except Exception as e:
                    print(f"Error loading label encoders from checkpoint: {e}")
        
        # If we have at least one label encoder, try to create default ones for missing
        if self.label_encoders:
            # Create default label encoders if missing
            default_categories = ['Road & Transportation', 'Water Leakage', 'Electricity', 'Waste Management', 'Other']
            default_subcategories = ['Pothole Repair', 'Pipeline Burst', 'Street Light Repair', 'Garbage Collection', 'Other']
            default_severities = ['Low', 'Medium', 'High']
            
            from sklearn.preprocessing import LabelEncoder
            
            if 'le_primary' not in self.label_encoders:
                le = LabelEncoder()
                le.classes_ = np.array(default_categories)
                self.label_encoders['le_primary'] = le
                print("✅ Created default le_primary")
            
            if 'le_subcat' not in self.label_encoders:
                le = LabelEncoder()
                le.classes_ = np.array(default_subcategories)
                self.label_encoders['le_subcat'] = le
                print("✅ Created default le_subcat")
            
            if 'le_severity' not in self.label_encoders:
                le = LabelEncoder()
                le.classes_ = np.array(default_severities)
                self.label_encoders['le_severity'] = le
                print("✅ Created default le_severity")
        
        return self.loaded_models
    
    def predict(self, text):
        """Predict using all loaded models with voting"""
        if not self.loaded_models:
            return None
        
        results = []
        
        for model_name in self.loaded_models:
            try:
                tokenizer = self.tokenizers.get(model_name)
                model = self.models.get(model_name)
                
                if tokenizer is None or model is None:
                    continue
                
                enc = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                    return_tensors="pt"
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                
                with torch.no_grad():
                    p1, p2, p3 = model(input_ids, attention_mask)
                    primary_pred = torch.argmax(p1, dim=1).item()
                    subcat_pred = torch.argmax(p2, dim=1).item()
                    severity_pred = torch.argmax(p3, dim=1).item()
                    primary_prob = torch.softmax(p1, dim=1).max().item()
                
                # Get label encoders
                le_primary = self.label_encoders.get('le_primary')
                le_subcat = self.label_encoders.get('le_subcat')
                le_severity = self.label_encoders.get('le_severity')
                
                if le_primary is None or le_subcat is None or le_severity is None:
                    continue
                
                primary_label = le_primary.inverse_transform([primary_pred])[0]
                subcat_label = le_subcat.inverse_transform([subcat_pred])[0]
                severity_label = le_severity.inverse_transform([severity_pred])[0]
                
                results.append({
                    'model': model_name,
                    'primary': primary_label,
                    'subcategory': subcat_label,
                    'severity': severity_label,
                    'confidence': primary_prob
                })
                
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                continue
        
        if results:
            # Voting
            primary_votes = {}
            subcat_votes = {}
            severity_votes = {}
            confidence_sum = 0
            
            for r in results:
                primary_votes[r['primary']] = primary_votes.get(r['primary'], 0) + 1
                subcat_votes[r['subcategory']] = subcat_votes.get(r['subcategory'], 0) + 1
                severity_votes[r['severity']] = severity_votes.get(r['severity'], 0) + 1
                confidence_sum += r['confidence']
            
            final_primary = max(primary_votes, key=primary_votes.get)
            final_subcat = max(subcat_votes, key=subcat_votes.get)
            final_severity = max(severity_votes, key=severity_votes.get)
            avg_confidence = confidence_sum / len(results)
            
            return {
                'primary': final_primary,
                'subcategory': final_subcat,
                'severity': final_severity,
                'confidence': avg_confidence,
                'all_predictions': results,
                'model_count': len(results),
                'vote_counts': {
                    'primary_votes': primary_votes,
                    'subcat_votes': subcat_votes,
                    'severity_votes': severity_votes
                }
            }
        
        return None


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load CSS
    load_css()
    
    # Initialize session state
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = ModelLoader()
        st.session_state.models_loaded = False
        st.session_state.prediction_result = None
        st.session_state.complaint_store = ComplaintStore()
        st.session_state.page = 'dashboard'
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h1>📋 Complaint Classification System</h1>
        <p>AI-Powered Multi-Model Complaint Analysis with Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/complaint.png", width=80)
        st.markdown("### ⚙️ Navigation")
        
        # Navigation buttons
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.page = 'dashboard'
            st.rerun()
        
        if st.button("📝 New Complaint", use_container_width=True):
            st.session_state.page = 'new_complaint'
            st.rerun()
        
        if st.button("📋 View All Complaints", use_container_width=True):
            st.session_state.page = 'view_complaints'
            st.rerun()
        
        st.markdown("---")
        
        # Model Management
        st.markdown("### 🤖 Model Management")
        
        if st.button("🚀 Load Models", use_container_width=True):
            with st.spinner("Loading models..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                progress_value = 0
                
                def update_progress(msg):
                    st.session_state.status_message = msg
                    status_text.text(msg)
                
                loaded = st.session_state.model_loader.load_models(update_progress)
                st.session_state.models_loaded = len(loaded) > 0
                
                progress_bar.progress(100)
                if loaded:
                    status_text.text(f"✅ Loaded {len(loaded)} models!")
                    st.success(f"Loaded: {', '.join(loaded)}")
                else:
                    status_text.text("❌ No models could be loaded")
                    st.error("No models loaded. Please check model files.")
                st.rerun()
        
        # Show loaded models
        if st.session_state.models_loaded:
            st.markdown("#### 📊 Loaded Models")
            for model_name in st.session_state.model_loader.loaded_models:
                config = ModelLoader.MODEL_CONFIGS.get(model_name, {})
                badge_class = config.get('badge', '')
                st.markdown(f'<span class="model-badge {badge_class}">{model_name}</span>', 
                           unsafe_allow_html=True)
            
            st.markdown(f"**Device:** {st.session_state.model_loader.device}")
            st.markdown(f"**Models Loaded:** {len(st.session_state.model_loader.loaded_models)}")
        else:
            st.warning("⚠️ No models loaded. Click 'Load Models' to load models.")
        
        # Stats
        st.markdown("---")
        stats = st.session_state.complaint_store.get_statistics()
        st.markdown("### 📈 Quick Stats")
        st.markdown(f"**Total Complaints:** {stats['total']}")
        st.markdown(f"✅ **Resolved:** {stats['resolved']}")
        st.markdown(f"⏳ **Pending:** {stats['pending']}")
        st.markdown(f"🔄 **In Progress:** {stats['in_progress']}")
        
        st.markdown("---")
        st.markdown("### 🏷️ Severity Levels")
        st.markdown("🔴 **High** - Immediate attention required")
        st.markdown("🟡 **Medium** - Timely resolution needed")
        st.markdown("🟢 **Low** - Regular course")
    
    # Page routing
    if st.session_state.page == 'dashboard':
        show_dashboard()
    elif st.session_state.page == 'new_complaint':
        show_new_complaint()
    elif st.session_state.page == 'view_complaints':
        show_view_complaints()
    elif st.session_state.page == 'update_complaint':
        show_update_complaint()

# ============================================================================
# DASHBOARD PAGE
# ============================================================================

def show_dashboard():
    st.markdown("### 📊 Complaint Dashboard")
    
    # Get statistics
    stats = st.session_state.complaint_store.get_statistics()
    df = st.session_state.complaint_store.get_all_complaints()
    
    # Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{stats['total']}</div>
            <div class="metric-label">Total Complaints</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-bottom-color: #28a745;">
            <div class="metric-value" style="color: #28a745;">{stats['resolved']}</div>
            <div class="metric-label">✅ Resolved</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-bottom-color: #ffc107;">
            <div class="metric-value" style="color: #ffc107;">{stats['pending']}</div>
            <div class="metric-label">⏳ Pending</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-bottom-color: #17a2b8;">
            <div class="metric-value" style="color: #17a2b8;">{stats['in_progress']}</div>
            <div class="metric-label">🔄 In Progress</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        resolution_rate = (stats['resolved'] / stats['total'] * 100) if stats['total'] > 0 else 0
        st.markdown(f"""
        <div class="metric-card" style="border-bottom-color: #6f42c1;">
            <div class="metric-value" style="color: #6f42c1;">{resolution_rate:.1f}%</div>
            <div class="metric-label">Resolution Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Category Distribution")
        if not df.empty:
            category_counts = df['primary_category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title='Complaints by Category',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No complaints data available")
    
    with col2:
        st.markdown("#### Status Distribution")
        if not df.empty:
            status_counts = df['status'].value_counts()
            colors = {'Resolved': '#28a745', 'Pending': '#ffc107', 'In Progress': '#17a2b8'}
            fig = px.bar(
                x=status_counts.index,
                y=status_counts.values,
                title='Complaints by Status',
                color=status_counts.index,
                color_discrete_map=colors
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No complaints data available")
    
    # Recent Complaints
    st.markdown("#### 📋 Recent Complaints")
    if not df.empty:
        display_df = df.sort_values('datetime', ascending=False).head(10)
        display_cols = ['id', 'name', 'title', 'primary_category', 'severity', 'status', 'datetime']
        st.dataframe(
            display_df[display_cols],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No complaints found")

    # Footer
    st.markdown("""
    <div class="footer">
        <p class="creator">👨‍💻 Created by: Shyam Tripathi (CH.SC.U4CSE23160) & Kothamasu Hridyaesh (CH.SC.U4CSE23117)</p>
        <p class="guide">📚 Guide: Dr. K. Ashwini (Assistant Professor Senior Grade)</p>
        <p style="color: #888; font-size: 0.9rem; margin-top: 0.5rem;">
            © 2026 Complaint Classification System | Multi-Model Ensemble
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# NEW COMPLAINT PAGE
# ============================================================================

def show_new_complaint():
    st.markdown("### 📝 File New Complaint")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        with st.form("complaint_form", clear_on_submit=False):
            st.markdown("#### 👤 Personal Information")
            name = st.text_input("Full Name *", placeholder="Enter your full name")
            location = st.text_input("Location *", placeholder="Enter your location")
            
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.text_input("Date & Time", value=current_datetime, disabled=True)
            
            st.markdown("---")
            st.markdown("#### 📄 Complaint Information")
            complaint_title = st.text_input("Complaint Title *", 
                                          placeholder="Brief title of your complaint")
            
            complaint_text = st.text_area(
                "Complaint Description *",
                placeholder="Describe your complaint in detail (English, Hindi, or Hinglish)...",
                height=150
            )
            
            st.caption("💡 You can write in English, Hindi, or Hinglish")
            st.caption("⚠️ * Required fields")
            
            submitted = st.form_submit_button("🔍 Analyze & Submit Complaint")
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        if submitted:
            if not name.strip():
                st.warning("⚠️ Please enter your name.")
            elif not location.strip():
                st.warning("⚠️ Please enter your location.")
            elif not complaint_text.strip():
                st.warning("⚠️ Please enter a complaint description.")
            elif not st.session_state.models_loaded:
                st.warning("⚠️ Please load models first from the sidebar.")
            else:
                with st.spinner("Analyzing complaint with all models..."):
                    full_text = f"{complaint_title} {complaint_text}" if complaint_title else complaint_text
                    result = st.session_state.model_loader.predict(full_text)
                    
                    if result:
                        # Save complaint
                        complaint_data = {
                            'name': name.strip(),
                            'location': location.strip(),
                            'datetime': current_datetime,
                            'title': complaint_title.strip() if complaint_title else '',
                            'description': complaint_text.strip(),
                            'primary_category': result['primary'],
                            'subcategory': result['subcategory'],
                            'severity': result['severity'],
                            'action_taken': 'Awaiting action',
                            'status': 'Pending',
                            'resolution_date': None,
                            'assigned_to': 'Not assigned'
                        }
                        complaint_id = st.session_state.complaint_store.add_complaint(complaint_data)
                        
                        # Display results
                        show_prediction_results(result, complaint_id)
                    else:
                        st.error("❌ Prediction failed. Please check if models are loaded.")
        
        # Check if we have a prediction result to display
        if 'show_result' in st.session_state and st.session_state.show_result:
            result = st.session_state.prediction_result
            complaint_id = st.session_state.complaint_id
            show_prediction_results(result, complaint_id)

def show_prediction_results(result, complaint_id):
    """Display prediction results"""
    
    st.markdown("#### ✅ Complaint Registered Successfully!")
    st.success(f"Complaint ID: **{complaint_id}**")
    
    # Result Cards
    col_pri, col_sub, col_sev = st.columns(3)
    
    with col_pri:
        st.markdown(f"""
        <div class="result-card" style="border-left-color: #1e3c72;">
            <h3 style="color: #1e3c72; font-size: 0.9rem;">📌 Primary Category</h3>
            <div style="font-size: 1.4rem; font-weight: 700; color: #1e3c72;">{result['primary']}</div>
            <div style="font-size: 0.9rem; color: #666;">Confidence: {result['confidence']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_sub:
        st.markdown(f"""
        <div class="result-card" style="border-left-color: #2a5298;">
            <h3 style="color: #2a5298; font-size: 0.9rem;">📂 Subcategory</h3>
            <div style="font-size: 1.4rem; font-weight: 700; color: #2a5298;">{result['subcategory']}</div>
            <div style="font-size: 0.9rem; color: #666;">Based on {result['model_count']} models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_sev:
        severity_emoji = "🔴" if result['severity'] == "High" else "🟡" if result['severity'] == "Medium" else "🟢"
        severity_color = "#dc3545" if result['severity'] == "High" else "#ffc107" if result['severity'] == "Medium" else "#28a745"
        st.markdown(f"""
        <div class="result-card" style="border-left-color: {severity_color};">
            <h3 style="color: {severity_color}; font-size: 0.9rem;">⚡ Severity</h3>
            <div style="font-size: 1.4rem; font-weight: 700; color: {severity_color};">{severity_emoji} {result['severity']}</div>
            <div style="font-size: 0.9rem; color: #666;">Priority Level</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Model-wise predictions
    st.markdown("#### 🤖 Model-wise Predictions")
    
    pred_data = []
    for pred in result['all_predictions']:
        pred_data.append({
            'Model': pred['model'],
            'Primary Category': pred['primary'],
            'Subcategory': pred['subcategory'],
            'Severity': pred['severity'],
            'Confidence': f"{pred['confidence']:.1%}"
        })
    
    df = pd.DataFrame(pred_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Clear result flag
    st.session_state.show_result = False
    
    # Button to view all complaints
    if st.button("📋 View All Complaints"):
        st.session_state.page = 'view_complaints'
        st.rerun()

# ============================================================================
# VIEW COMPLAINTS PAGE
# ============================================================================

def show_view_complaints():
    st.markdown("### 📋 All Complaints")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ['All', 'Pending', 'In Progress', 'Resolved']
        )
    with col2:
        severity_filter = st.selectbox(
            "Filter by Severity",
            ['All', 'High', 'Medium', 'Low']
        )
    with col3:
        category_filter = st.selectbox(
            "Filter by Category",
            ['All'] + list(st.session_state.complaint_store.get_statistics()['categories'].keys())
        )
    
    # Get filtered data
    df = st.session_state.complaint_store.get_all_complaints()
    
    if df.empty:
        st.info("No complaints found.")
        return
    
    # Apply filters
    if status_filter != 'All':
        df = df[df['status'] == status_filter]
    if severity_filter != 'All':
        df = df[df['severity'] == severity_filter]
    if category_filter != 'All':
        df = df[df['primary_category'] == category_filter]
    
    # Display complaints
    st.markdown(f"**Showing {len(df)} complaints**")
    
    # Create display dataframe
    display_cols = ['id', 'name', 'location', 'datetime', 'title', 
                    'primary_category', 'subcategory', 'severity', 'status', 'action_taken']
    
    # Add action buttons
    if not df.empty:
        # Use columns layout for each complaint
        for idx, row in df.iterrows():
            with st.expander(f"{row['id']} - {row['title']} ({row['status']})"):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Name:** {row['name']}")
                    st.markdown(f"**Location:** {row['location']}")
                    st.markdown(f"**Date:** {row['datetime']}")
                    st.markdown(f"**Category:** {row['primary_category']}")
                    st.markdown(f"**Subcategory:** {row['subcategory']}")
                    severity_emoji = "🔴" if row['severity'] == "High" else "🟡" if row['severity'] == "Medium" else "🟢"
                    st.markdown(f"**Severity:** {severity_emoji} {row['severity']}")
                    st.markdown(f"**Description:** {row['description'][:200]}..." if len(row['description']) > 200 else f"**Description:** {row['description']}")
                    st.markdown(f"**Action Taken:** {row['action_taken']}")
                    if row['resolution_date']:
                        st.markdown(f"**Resolved On:** {row['resolution_date']}")
                
                with col2:
                    # Status update form
                    with st.form(f"update_form_{row['id']}"):
                        st.markdown("#### Update Status")
                        new_status = st.selectbox(
                            "Status",
                            ['Pending', 'In Progress', 'Resolved'],
                            index=['Pending', 'In Progress', 'Resolved'].index(row['status']) if row['status'] in ['Pending', 'In Progress', 'Resolved'] else 0,
                            key=f"status_{row['id']}"
                        )
                        action_taken = st.text_input(
                            "Action Taken",
                            value=row['action_taken'] if row['action_taken'] != 'Awaiting action' else '',
                            key=f"action_{row['id']}"
                        )
                        assigned_to = st.text_input(
                            "Assigned To",
                            value=row['assigned_to'] if row['assigned_to'] != 'Not assigned' else '',
                            key=f"assigned_{row['id']}"
                        )
                        
                        if st.form_submit_button("Update", key=f"update_btn_{row['id']}"):
                            success = st.session_state.complaint_store.update_status(
                                row['id'],
                                new_status,
                                action_taken if action_taken else 'No action recorded',
                                assigned_to if assigned_to else 'Not assigned'
                            )
                            if success:
                                st.success(f"✅ Complaint {row['id']} updated successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to update complaint.")


def show_update_complaint():
    st.markdown("### 🔄 Update Complaint")
    st.info("Select a complaint from the 'View All Complaints' page to update its status.")

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()