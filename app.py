"""
Model loading and management utilities.
"""
import torch
import streamlit as st
from pathlib import Path

from model import CNN
from config import MODEL_PATH


@st.cache_resource
def load_model(model_path: Path = MODEL_PATH) -> torch.nn.Module:
    """
    Load the trained model from disk with caching.
    
    Args:
        model_path: Path to the model checkpoint file
        
    Returns:
        torch.nn.Module: Loaded model in evaluation mode
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Please ensure the model file is in the correct location."
        )
    
    try:
        model = CNN()
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def get_model_info(model: torch.nn.Module) -> dict:
    """
    Get information about the loaded model.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Model information including parameter count
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_type": model.__class__.__name__
    }
