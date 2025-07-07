import gradio as gr
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import pandas as pd

# Load model and tokenizer
model_name = "distilbert-base-uncased"
model = AutoModel.from_pretrained(model_name, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def create_attention_matrix_html(tokens, attention_matrix):
    """Create an HTML table showing the attention matrix"""
    # Clean tokens for display (remove ##)
    display_tokens = [token.replace("##", "") for token in tokens]
    
    html = "<table style='border-collapse: collapse; margin: 0; font-size: 11px; width: 100%;'>"
    
    # Header row
    html += "<tr><th style='border: 1px solid #ddd; padding: 3px; background-color: #f2f2f2; font-size: 10px;'>From\\To</th>"
    for token in display_tokens:
        html += f"<th style='border: 1px solid #ddd; padding: 3px; background-color: #f2f2f2; font-size: 9px; max-width: 40px; word-break: break-all;'>{token}</th>"
    html += "</tr>"
    
    # Data rows
    for i, from_token in enumerate(display_tokens):
        html += "<tr>"
        html += f"<td style='border: 1px solid #ddd; padding: 3px; font-weight: bold; background-color: #f9f9f9; font-size: 9px; max-width: 50px; word-break: break-all;'>{from_token}</td>"
        
        for j in range(len(display_tokens)):
            weight = attention_matrix[i][j]
            # Color intensity based on weight
            intensity = min(weight * 3, 1.0)  # Scale for better visibility
            color = f"rgba(255, 99, 71, {intensity})"
            html += f"<td style='border: 1px solid #ddd; padding: 2px; text-align: center; background-color: {color}; font-size: 8px;'>{weight:.3f}</td>"
        
        html += "</tr>"
    
    html += "</table>"
    return html

def extract_bert_attention(text, layer=2, head=0, focus_token_idx=1):
    """Extract and visualize BERT attention with step-by-step calculations"""
    try:
        # Step 1: Tokenization
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=50)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Step 2: Model forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            attention_tensor = outputs.attentions[layer][0][head].numpy()
        
        # Ensure focus token is valid
        if focus_token_idx >= len(tokens):
            focus_token_idx = 1 if len(tokens) > 1 else 0
        
        # Step 3: Extract attention weights for focus token
        attention_weights = attention_tensor[focus_token_idx]
        
        # Clean display tokens
        display_tokens = [token.replace("##", "") for token in tokens]
        
        # Get focus token name first
        focus_token_name = display_tokens[focus_token_idx] if focus_token_idx < len(display_tokens) else "N/A"
        
        # Create compact calculation steps with real model information
        d_k = 64  # DistilBERT head dimension  
        scaling_factor = np.sqrt(d_k)
        
        steps_html = f"""
<div style="background: #f8f9fa; padding: 15px; border-radius: 6px; margin: 0; border: 1px solid #e9ecef; font-size: 14px;">
    <h4 style="color: #495057; margin: 0 0 15px 0; font-size: 16px;">🧮 Self-Attention Calculation Steps</h4>
    
    <div style="background: white; padding: 12px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #007bff;">
        <h5 style="color: #007bff; margin: 0 0 8px 0;">Step 1: Input Processing</h5>
        <p style="margin: 5px 0;"><strong>Input text:</strong> "{text}"</p>
        <p style="margin: 5px 0;"><strong>Tokenization:</strong> {len(tokens)} tokens → {display_tokens}</p>
        <p style="margin: 5px 0; color: #666; font-size: 13px;"><em>Why:</em> BERT breaks text into subwords it recognizes from training.</p>
    </div>
    
    <div style="background: white; padding: 12px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #28a745;">
        <h5 style="color: #28a745; margin: 0 0 8px 0;">Step 2: Embeddings</h5>
        <p style="margin: 5px 0;"><strong>Each token → 768-dimensional vector</strong></p>
        <p style="margin: 5px 0;">Example: "{focus_token_name}" becomes [0.12, -0.45, 0.78, ..., 0.33] (768 numbers)</p>
        <p style="margin: 5px 0; color: #666; font-size: 13px;"><em>Why:</em> Converts words to numbers that capture semantic meaning.</p>
    </div>
    
    <div style="background: white; padding: 12px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #dc3545;">
        <h5 style="color: #dc3545; margin: 0 0 8px 0;">Step 3: Query, Key, Value Creation</h5>
        <p style="margin: 5px 0;"><strong>Query (Q):</strong> "What is this token looking for?"</p>
        <p style="margin: 5px 0;"><strong>Key (K):</strong> "What does this token offer?"</p>
        <p style="margin: 5px 0;"><strong>Value (V):</strong> "What information does this token contain?"</p>
        <p style="margin: 8px 0; font-family: monospace; background: #f8f9fa; padding: 5px;">
            Q = Embedding × W<sub>Q</sub> &nbsp;&nbsp; (768 × 64 = 64 dimensions)<br>
            K = Embedding × W<sub>K</sub> &nbsp;&nbsp; (768 × 64 = 64 dimensions)<br>
            V = Embedding × W<sub>V</sub> &nbsp;&nbsp; (768 × 64 = 64 dimensions)
        </p>
        <p style="margin: 5px 0; color: #666; font-size: 13px;"><em>Why:</em> Creates specialized vectors for matching (Q×K) and information (V).</p>
    </div>
    
    <div style="background: white; padding: 12px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #fd7e14;">
        <h5 style="color: #fd7e14; margin: 0 0 8px 0;">Step 4: Attention Score Calculation</h5>
        <p style="margin: 5px 0;"><strong>For token "{focus_token_name}":</strong></p>
        <p style="margin: 8px 0; font-family: monospace; background: #f8f9fa; padding: 8px;">
            Score = Q<sub>{focus_token_name}</sub> · K<sub>each_token</sub><br>
            Raw scores: [calculated by dot product]<br>
            Scaled scores: Raw ÷ √64 = Raw ÷ {scaling_factor:.1f}
        </p>
        <p style="margin: 5px 0; color: #666; font-size: 13px;"><em>Why:</em> Measures compatibility. Scaling prevents values from getting too large.</p>
    </div>
    
    <div style="background: white; padding: 12px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #6f42c1;">
        <h5 style="color: #6f42c1; margin: 0 0 8px 0;">Step 5: Softmax Normalization</h5>
        <p style="margin: 8px 0; font-family: monospace; background: #f8f9fa; padding: 8px;">
            Attention<sub>i</sub> = exp(score<sub>i</sub>) ÷ Σ exp(score<sub>j</sub>)<br>
            Result: All weights sum to 1.0 (probability distribution)
        </p>
        <p style="margin: 5px 0; color: #666; font-size: 13px;"><em>Why:</em> Converts raw scores to probabilities that sum to 1.</p>
    </div>
    
    <div style="background: white; padding: 12px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #20c997;">
        <h5 style="color: #20c997; margin: 0 0 8px 0;">Step 6: Final Output</h5>
        <p style="margin: 8px 0; font-family: monospace; background: #f8f9fa; padding: 8px;">
            Context<sub>{focus_token_name}</sub> = Σ (Attention<sub>i</sub> × Value<sub>i</sub>)
        </p>
        <p style="margin: 5px 0;"><strong>Result:</strong> A new vector that combines information from all tokens, weighted by attention.</p>
        <p style="margin: 5px 0; color: #666; font-size: 13px;"><em>Why:</em> Creates context-aware representation of "{focus_token_name}".</p>
    </div>
    
    <div style="background: #e8f5e8; padding: 12px; border-radius: 5px; margin: 15px 0; border: 1px solid #c3e6c3;">
        <h5 style="margin: 0 0 8px 0; color: #2d5a2d;">🎯 Actual Results from DistilBERT:</h5>
        <p style="margin: 5px 0;"><strong>Focus token:</strong> "{focus_token_name}" (position {focus_token_idx})</p>
        <p style="margin: 5px 0;"><strong>Layer {layer+1}, Head {head+1}</strong></p>
        <div style="margin: 8px 0;">
            <strong>Top attention weights:</strong>
"""
        
        # Show top 4 attention weights with more detail
        top_indices = np.argsort(attention_weights)[-4:][::-1]
        for idx in top_indices:
            if idx < len(display_tokens):
                percentage = attention_weights[idx] * 100
                steps_html += f"            <div style='margin: 3px 0; padding: 3px; background: rgba(40, 167, 69, {attention_weights[idx]});'>"
                steps_html += f"{display_tokens[idx]}: {attention_weights[idx]:.4f} ({percentage:.1f}%)</div>"
        
        steps_html += f"""
        </div>
        <p style="margin: 8px 0; font-weight: bold;">✓ Verification: Sum = {sum(attention_weights):.4f} ≈ 1.0</p>
    </div>
    
    <div style="background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; border: 1px solid #ffeaa7;">
        <h5 style="margin: 0 0 5px 0; color: #856404;">💡 Key Insight:</h5>
        <p style="margin: 0; font-size: 13px;">
            Token "{focus_token_name}" pays most attention to the tokens shown above. 
            This means when processing "{focus_token_name}", the model heavily considers 
            the context and meaning of these related tokens.
        </p>
    </div>
</div>
"""
        
        # Focus token analysis (show ALL tokens)
        focus_analysis = f"""
<div style="background: #e3f2fd; padding: 10px; border-radius: 6px; margin: 0; border: 1px solid #bbdefb; height: 300px; overflow-y: auto;">
    <h4 style="color: #1976d2; margin: 0 0 8px 0;">🎯 Focus Token: "{focus_token_name}" (pos {focus_token_idx})</h4>
    <div style="font-size: 11px; line-height: 1.4;">
        <p style="margin-bottom: 8px;"><strong>Attention to all tokens:</strong></p>
"""
        
        # Show ALL tokens with their weights
        for i, (token, weight) in enumerate(zip(display_tokens, attention_weights)):
            focus_analysis += f"        <div><strong>{token}:</strong> {weight:.4f} ({weight*100:.1f}%)</div>\n"
        
        focus_analysis += f"""
    </div>
    <p style="margin-top: 8px; font-size: 11px; color: #666; font-weight: bold;">Sum: {sum(attention_weights):.4f} | Layer {layer+1}, Head {head+1}</p>
</div>
"""
        
        # Full matrix display - no scrollbars, pushed to top
        matrix_html = create_attention_matrix_html(tokens, attention_tensor)
        matrix_section = f"""
<style>
.gradio-container .block {{
    margin: 0 !important;
    padding: 0 !important;
}}
.gradio-container .form {{
    gap: 0 !important;
}}
</style>
<div style="margin: 0 !important; padding: 0 !important;">
    <h4 style="color: #495057; margin: 0 !important; padding: 3px 0; font-size: 14px;">📊 Attention Matrix</h4>
    <div style="margin: 0 !important; padding: 0 !important; overflow: visible;">
        {matrix_html}
    </div>
</div>
"""
        
        return {
            "steps": steps_html,
            "focus": focus_analysis, 
            "matrix": matrix_section
        }
        
    except Exception as e:
        error_msg = f"<div style='color: red; padding: 10px;'>Error: {str(e)}</div>"
        return {
            "steps": error_msg,
            "focus": "",
            "matrix": ""
        }

# Create 2x2 grid interface
with gr.Blocks(title="Self-Attention Live Demo", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# Self-Attention Live Demo")
    gr.Markdown("Real BERT attention weights with step-by-step calculations")
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Enter text to analyze",
                value="The animal didn't cross the street because it was too scared",
                lines=3
            )
            layer_slider = gr.Slider(0, 5, value=2, step=1, label="Layer")
            head_slider = gr.Slider(0, 11, value=0, step=1, label="Head") 
            focus_slider = gr.Slider(1, 15, value=1, step=1, label="Focus Token Index")
            
            analyze_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column(scale=1):
            focus_output = gr.HTML(label="Focus Token Analysis")
    
    with gr.Row():
        with gr.Column(scale=1):
            steps_output = gr.HTML(label="Calculation Steps")
        
        with gr.Column(scale=1):
            matrix_output = gr.HTML(label="Attention Matrix")
    
    def process_attention(text, layer, head, focus_idx):
        result = extract_bert_attention(text, layer, head, focus_idx)
        return result["focus"], result["steps"], result["matrix"]
    
    analyze_btn.click(
        fn=process_attention,
        inputs=[text_input, layer_slider, head_slider, focus_slider],
        outputs=[focus_output, steps_output, matrix_output]
    )
    
    # Auto-run on load
    interface.load(
        fn=process_attention,
        inputs=[text_input, layer_slider, head_slider, focus_slider],
        outputs=[focus_output, steps_output, matrix_output]
    )

# Launch with public sharing enabled
if __name__ == "__main__":
    interface.launch(
        share=False,  # HF Spaces handles sharing
        server_name="0.0.0.0",
        server_port=7860
    )
