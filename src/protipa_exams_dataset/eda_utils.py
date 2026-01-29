import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from PIL import Image
import textwrap
from IPython.display import display, Markdown, HTML

import base64
import seaborn as sns

# Set a premium aesthetic for all plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'sans-serif'

def _get_image(img_data):
    """
    Helper to convert image data (which might be a dict from HF datasets) 
    into a format matplotlib or PIL can handle.
    """
    if isinstance(img_data, dict):
        if 'bytes' in img_data and img_data['bytes'] is not None:
            return Image.open(io.BytesIO(img_data['bytes']))
        elif 'path' in img_data and img_data['path'] is not None:
            return Image.open(img_data['path'])
    return img_data

def _process_text(text):
    """
    Cleans up text for HTML display, handling various newline representations and LaTeX.
    Uses <br> for line breaks.
    """
    if not isinstance(text, str):
        return str(text)
    
    # Replace literal newlines and real newlines with <br>
    # Start with most escaped to least escaped
    text = text.replace('\\\\\\\\n', '<br>').replace('\\\\n', '<br>').replace('\\n', '<br>')
    text = text.replace('\r\n', '<br>').replace('\n', '<br>')
    
    return text

def _img_to_base64_html(img_data, max_width=300):
    """
    Converts image data to a base64 encoded HTML img tag.
    """
    try:
        if isinstance(img_data, dict) and img_data.get('bytes'):
            b64_str = base64.b64encode(img_data['bytes']).decode('utf-8')
            return f'<div style="text-align: center; margin: 10px 0;"><img src="data:image/png;base64,{b64_str}" style="max-width: {max_width}px; height: auto; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></div>'
        elif isinstance(img_data, Image.Image):
            buffered = io.BytesIO()
            img_data.save(buffered, format="PNG")
            b64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return f'<div style="text-align: center; margin: 10px 0;"><img src="data:image/png;base64,{b64_str}" style="max-width: {max_width}px; height: auto; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);"></div>'
    except Exception:
        return "[Image]"
    return ""

def _get_mathjax_trigger(container_id):
    """
    Returns a script block to trigger MathJax typesetting with $ support.
    """
    return f"""
    <script>
        (function() {{
            function configureAndTrigger() {{
                if (window.MathJax) {{
                    // v2 Config
                    if (MathJax.Hub) {{
                        MathJax.Hub.Config({{
                            tex2jax: {{
                                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                                displayMath: [['$$', '$$'], ['\\\\[', ' \\\\]']],
                                processEscapes: true
                            }}
                        }});
                        MathJax.Hub.Queue(["Typeset", MathJax.Hub, '{container_id}']);
                    }}
                    // v3 Config
                    if (MathJax.typesetPromise) {{
                        if (MathJax.config && MathJax.config.tex) {{
                            MathJax.config.tex.inlineMath = [['$', '$'], ['\\\\(', '\\\\)']];
                        }}
                        MathJax.typesetPromise([document.getElementById('{container_id}')]).catch(function (err) {{
                            console.log('MathJax typeset failed: ' + err.message);
                        }});
                    }}
                }}
            }}
            
            // Load MathJax if totally missing
            if (!window.MathJax) {{
                var script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
                script.async = true;
                script.onload = configureAndTrigger;
                document.head.appendChild(script);
                window.MathJax = {{
                    tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }}
                }};
            }} else {{
                configureAndTrigger();
            }}

            // Multiple retries for different loading stages
            setTimeout(configureAndTrigger, 100); 
            setTimeout(configureAndTrigger, 1000);
            setTimeout(configureAndTrigger, 3000);
        }})();
    </script>
    """

def display_samples(df, n=3, prioritize_images=False):
    """
    Selects n random samples and displays them as an HTML table with LaTeX support and embedded images.
    """
    # Sampling logic
    if prioritize_images:
        has_images = df['images'].apply(lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False)
        priority_samples = df[has_images]
        if len(priority_samples) >= n:
            samples = priority_samples.sample(n)
        else:
            samples = df.sample(min(n, len(df)))
    else:
        samples = df.sample(min(n, len(df)))

    cols = ['id', 'subject', 'year', 'admission_level', 'question', 'input', 'image', 'choices', 'answer', 'answer_text', 'answer_index']
    
    # Add embedded images column if 'images' exists
    if 'images' in samples.columns:
        # Create a copy to avoid SettingWithCopyWarning
        samples = samples.copy()
        samples['image'] = samples['images'].apply(lambda imgs: _img_to_base64_html(imgs[0], 150) if isinstance(imgs, (list, np.ndarray)) and len(imgs) > 0 else "")
    
    available_cols = [c for c in cols if c in samples.columns]
    
    # Create HTML table
    display_df = samples[available_cols].copy()
    
    # Format answer_index as integer if it exists and is not null
    if 'answer_index' in display_df.columns:
        display_df['answer_index'] = display_df['answer_index'].apply(lambda x: int(x) if pd.notna(x) else "")
    
    # Process text columns
    for col in ['question', 'answer_text', 'input']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: _process_text(x) if pd.notna(x) else "")
            
    # Format choices for table
    if 'choices' in display_df.columns:
        display_df['choices'] = display_df['choices'].apply(
            lambda x: "<br>".join([f"• {_process_text(c)}" for c in x]) if isinstance(x, (list, np.ndarray)) and len(x) > 0 else ""
        )
    
    html_out = display_df.to_html(escape=False, index=False, classes='dataframe styled-table')
    
    import uuid
    container_id = f"table-container-{uuid.uuid4().hex[:8]}"
    
    styled_html = f"""
    <style>
        .styled-table {{
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.95em;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            width: 100%;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.08);
            border-radius: 8px;
            overflow: hidden;
        }}
        .styled-table thead tr {{
            background-color: #009879;
            color: #ffffff;
            text-align: left;
            font-weight: bold;
        }}
        .styled-table th,
        .styled-table td {{
            padding: 12px 15px;
            text-align: left !important;
            vertical-align: top !important;
            white-space: pre-wrap !important;
            word-wrap: break-word !important;
            max-width: 400px;
        }}
        .styled-table tbody tr {{
            border-bottom: 1px solid #eeeeee;
        }}
        .styled-table tbody tr:nth-of-type(even) {{
            background-color: #fcfcfc;
        }}
        .styled-table tbody tr:last-of-type {{
            border-bottom: 2px solid #009879;
        }}
        .styled-table tbody tr:hover {{
            background-color: #f8f8f8;
        }}
        .styled-table .id-cell {{ font-family: monospace; font-weight: bold; color: #555; }}
    </style>
    <div id='{container_id}' style='overflow-x:auto;'>{html_out}</div>
    {_get_mathjax_trigger(container_id)}
    """
    display(HTML(styled_html))

def print_sample(df, sample_id):
    """
    Prints detailed information about a single question using integrated HTML styling and images.
    """
    if isinstance(sample_id, int):
        row = df.iloc[sample_id]
    else:
        filtered = df[df['id'] == sample_id]
        if filtered.empty:
            print(f"Sample with ID {sample_id} not found.")
            return
        row = filtered.iloc[0]

    import uuid
    container_id = f"sample-container-{uuid.uuid4().hex[:8]}"

    # Prepare images as embedded HTML
    img_html = ""
    if 'images' in row and row['images'] is not None and len(row['images']) > 0:
        img_html = '<div style="margin-top: 20px; display: flex; flex-direction: column; align-items: center; gap: 15px;">'
        for img_data in row['images']:
            img_html += _img_to_base64_html(img_data, 600)
        img_html += "</div>"

    choices_section = ""
    if 'choices' in row and isinstance(row['choices'], (list, np.ndarray)) and len(row['choices']) > 0:
        choices_list = "".join([f"<li style='margin-bottom: 12px; list-style-type: none; border: 1px solid #f0f0f0; padding: 10px 15px; border-radius: 6px; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.02);'>{_process_text(c)}</li>" for c in row['choices']])
        choices_section = f"""
        <div style="margin: 25px 0;">
            <p style="font-weight: bold; color: #2c3e50; font-size: 1.1em; margin-bottom: 15px;">Options:</p>
            <ul style="padding-left: 0;">{choices_list}</ul>
        </div>
        """

    input_section = ""
    if 'input' in row and row['input'] and str(row['input']).strip():
        input_section = f"""
        <div style="margin-bottom: 25px; background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 5px solid #95a5a6; border-right: 1px solid #e9ecef; border-top: 1px solid #e9ecef; border-bottom: 1px solid #e9ecef;">
            <p style="font-weight: 700; color: #7f8c8d; margin-bottom: 12px; font-size: 1.1em; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.8;">Context / Input</p>
            <div style="font-size: 1.1em; color: #34495e; white-space: pre-wrap; font-style: italic;">{_process_text(row['input'])}</div>
        </div>
        """

    header_html = f"""
    <div id="{container_id}" style="max-width: 950px; border: 1px solid #e1e4e8; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 24px rgba(0,0,0,0.08); margin: 25px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; background-color: #fff;">
        <div style="background-color: #24292e; color: #fff; padding: 20px 30px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h2 style="margin: 0; font-size: 1.5em; color: #fff; letter-spacing: -0.5px;">{row['id']}</h2>
                <div style="margin-top: 6px; opacity: 0.8; font-size: 0.9em; font-weight: 500;">{row['subject'].upper()} • {row['year']} • {row['admission_level'].upper()}</div>
            </div>
            <div style="background: rgba(255,255,255,0.15); padding: 5px 15px; border-radius: 20px; font-size: 0.85em; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{row['format']}</div>
        </div>
        
        <div style="padding: 30px; border-bottom: 1px solid #f0f0f0;">
             <div style="margin-bottom: 25px;">
                <p style="font-weight: 700; color: #1a1a1a; margin-bottom: 12px; font-size: 1.1em; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.7;">Question</p>
                <div style="font-size: 1.15em; color: #24292e; white-space: pre-wrap; background: #fdfdfd; padding: 20px; border-radius: 8px; border-left: 5px solid #3498db; box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);">{_process_text(row['question'])}</div>
            </div>

            {input_section}
            {img_html}
            {choices_section}

            <div style="margin-top: 35px; border-top: 2px dashed #eee; padding-top: 25px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                    <span style="background: #2ecc71; color: white; padding: 5px 15px; border-radius: 6px; font-weight: 800; font-size: 0.75em; text-transform: uppercase; letter-spacing: 1px;">Correct Answer</span>
                </div>
                <div style="font-size: 1.15em; font-weight: 500; color: #27ae60; background: #f9fffb; padding: 20px; border-radius: 8px; border-left: 5px solid #2ecc71; white-space: pre-wrap; display: flex; justify-content: space-between; align-items: center;">
                    <span>{_process_text(row['answer_text'])}</span>
                    {f'<span style="background: #e1f7e7; color: #27ae60; padding: 5px 10px; border-radius: 4px; font-weight: bold; font-family: monospace; font-size: 0.8em; margin-left: 10px;">ID: {int(row["answer_index"])}</span>' if "answer_index" in row and pd.notna(row["answer_index"]) else (f'<span style="background: #e1f7e7; color: #27ae60; padding: 5px 10px; border-radius: 4px; font-weight: bold; font-family: monospace; font-size: 0.8em; margin-left: 10px;">ID: {row["answer"]}</span>' if "answer" in row and pd.notna(row["answer"]) else "")}
                </div>
            </div>
        </div>
        {_get_mathjax_trigger(container_id)}
    </div>
    """
    display(HTML(header_html))
    print("\n" + "=" * 60 + "\n")

def get_image_samples_df(df, n=5):
    """
    Returns a subset of the dataframe containing questions with images.
    """
    mask = df['images'].apply(lambda x: len(x) > 0 if x is not None else False)
    df_with_images = df[mask]
    
    if df_with_images.empty:
        return pd.DataFrame()
    
    return df_with_images.sample(min(n, len(df_with_images)))

def display_multimodal_samples(df, n=3):
    """
    Displays samples that contain images using the integrated table display.
    """
    samples = get_image_samples_df(df, n=n)
    
    if samples.empty:
        print("No samples with images found.")
        return

    print(f"Showing {len(samples)} samples with images:")
    # We pass the subset to display_samples but since display_samples itself samples,
    # we just pass all of them and set n to the count.
    display_samples(samples, n=len(samples))

def plot_subject_dist(df, show_plot=False):
    """
    Plots and returns subject distribution and its frequency table.
    """
    counts = df['subject'].value_counts()
    freq_df = counts.reset_index()
    freq_df.columns = ['Subject', 'Count']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df, 
        x='subject', 
        order=counts.index, 
        hue='subject', 
        palette="viridis", 
        legend=False, 
        ax=ax
    )
    ax.set_title('Questions per Subject')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig, freq_df

def plot_level_dist(df, show_plot=False):
    """
    Plots and returns admission level distribution and its frequency table.
    """
    counts = df['admission_level'].value_counts()
    freq_df = counts.reset_index()
    freq_df.columns = ['Admission Level', 'Count']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df, 
        x='admission_level', 
        hue='admission_level', 
        palette="magma", 
        legend=False, 
        ax=ax
    )
    ax.set_title('Questions per Level')
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig, freq_df

def plot_format_dist(df, show_plot=False):
    """
    Plots and returns question format distribution and its frequency table.
    """
    counts = df['format'].value_counts()
    freq_df = counts.reset_index()
    freq_df.columns = ['Format', 'Count']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df, 
        x='format', 
        order=counts.index, 
        hue='format', 
        palette="coolwarm", 
        legend=False, 
        ax=ax
    )
    ax.set_title('Question Formats')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig, freq_df

def plot_reference_dist(df, show_plot=False):
    """
    Plots and returns reference type distribution and its frequency table.
    """
    counts = df['reference'].value_counts()
    freq_df = counts.reset_index()
    freq_df.columns = ['Reference', 'Count']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df, 
        x='reference', 
        order=counts.index, 
        hue='reference', 
        palette="Set2", 
        legend=False, 
        ax=ax
    )
    ax.set_title('Reference Types')
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig, freq_df

def plot_temporal_distribution(df, show_plot=False):
    """
    Plots and returns temporal distribution and its frequency table.
    """
    year_counts = df.groupby(['year', 'subject']).size().unstack().fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    year_counts.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Dataset Growth/Coverage over Years')
    ax.set_ylabel('Question Count')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig, year_counts

def plot_points_distribution(df, show_plot=False):
    """
    Calculates, plots, and returns point value distribution and its summary table.
    """
    print("Percentage of questions with point values assigned:")
    points_presence = df['points'].notna().mean() * 100
    print(f"{points_presence:.2f}%")

    stats_df = df['points'].describe().to_frame().transpose()
    
    fig = None
    if points_presence > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df['points'].dropna(), bins=20, kde=True, color="skyblue", ax=ax)
        ax.set_title('Distribution of Question Point Values')
        plt.tight_layout()
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
    
    return fig, stats_df

def plot_format_by_subject(df, show_plot=False):
    """
    Plots the question format distribution grouped by subject.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate counts and pivot for grouped bar chart
    format_counts = df.groupby(['subject', 'format']).size().unstack(fill_value=0)
    
    format_counts.plot(kind='bar', ax=ax)
    
    ax.set_title('Question Format Distribution by Subject')
    ax.set_ylabel('Count')
    ax.set_xlabel('Subject')
    plt.xticks(rotation=45)
    plt.legend(title='Format', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig, format_counts

def plot_points_by_level(df, show_plot=False):
    """
    Plots the point distribution grouped by admission level (gymnasium vs lyceum).
    """
    df_points = df[df['points'].notna()]
    
    if df_points.empty:
        print("No point values available to plot.")
        return None, None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.boxplot(data=df_points, x='admission_level', y='points', hue='admission_level', palette="Set2", ax=ax)
    
    ax.set_title('Question Point Values by Admission Level')
    ax.set_ylabel('Points')
    ax.set_xlabel('Admission Level')
    
    summary_stats = df_points.groupby('admission_level')['points'].describe()
    
    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    return fig, summary_stats
