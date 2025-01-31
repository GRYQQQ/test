#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install scikit-learn


# In[2]:


#pip install pypdf


# In[3]:


#pip install python-docx


# In[4]:


#pip install ipywidgets


# In[5]:


#pip install widgetsnbextension


# In[6]:


#!jupyter nbextension enable --py widgetsnbextension --sys-prefix
#!jupyter serverextension enable voila --sys-prefix


# In[7]:


import os
import pandas as pd
from ipywidgets import widgets, VBox, FileUpload, Button, HBox
from IPython.display import display, clear_output


# In[8]:


directory_path = os.getcwd()
uploaded_df = pd.DataFrame(columns=['Document Name', 'Uploaded By']) 
output = widgets.Output()  


# In[9]:


# Function to list existing documents
def list_existing_documents(folder):
    global uploaded_df
    documents = [file for file in os.listdir(folder) if file.endswith(('.pdf', '.docx'))]
    
    if uploaded_df.empty:
        uploaded_df = pd.DataFrame({'Document Name': documents, 'Uploaded By': ['Unknown'] * len(documents)})
    else:
        new_files = [file for file in documents if file not in uploaded_df['Document Name'].values]
        if new_files:
            new_rows = pd.DataFrame({'Document Name': new_files, 'Uploaded By': ['Unknown'] * len(new_files)})
            uploaded_df = pd.concat([uploaded_df, new_rows], ignore_index=True)
    return uploaded_df

# Function to update the displayed table
def update_table():
    with output:
        output.clear_output()
        display(uploaded_df)

# Load existing documents and update table
list_existing_documents(directory_path)
update_table()

# File upload widgets
upload_widget = FileUpload(accept=".pdf, .docx", multiple=True)
username_input = widgets.Text(description="Uploaded By:", placeholder="Enter your name")
upload_button = Button(description="Upload Files", button_style='success')

# Function to handle file upload
def on_upload_click(b):
    global uploaded_df
    if not username_input.value.strip():
        with output:
            output.clear_output()
            print("Please enter your name before uploading!")
        return
    
    for fileinfo in upload_widget.value:
        filename = fileinfo['name']
        filepath = os.path.join(directory_path, filename)
        
        # Check if file already exists
        if filename in uploaded_df['Document Name'].values:
            with output:
                output.clear_output()
                print(f"File '{filename}' already exists! Please upload a different file.")
            return
        
        # Save the file
        try:
            with open(filepath, 'wb') as f:
                f.write(fileinfo['content'])
            new_row = {'Document Name': filename, 'Uploaded By': username_input.value.strip()}
            uploaded_df = pd.concat([uploaded_df, pd.DataFrame([new_row])], ignore_index=True)
        except Exception as e:
            with output:
                output.clear_output()
                print(f"Failed to upload file '{filename}': {str(e)}")
            return
    
    with output:
        clear_output()
        print("Files uploaded successfully!")
        display(uploaded_df)

# Bind upload button click event
upload_button.on_click(on_upload_click)

# Function to delete a file
def delete_file(filename):
    global uploaded_df
    filepath = os.path.join(directory_path, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        uploaded_df = uploaded_df[uploaded_df['Document Name'] != filename]
        with output:
            clear_output()
            print(f"File '{filename}' deleted successfully!")
            display(uploaded_df)
    else:
        with output:
            clear_output()
            print(f"File '{filename}' does not exist!")

# Function to create a delete button for each file
def create_delete_button(filename):
    delete_button = Button(description=f"Delete {filename}", button_style='danger')
    
    def on_delete_click(b):
        delete_file(filename)
    
    delete_button.on_click(on_delete_click)
    return delete_button

# Function to update the table with delete buttons
def update_table_with_buttons():
    with output:
        output.clear_output()
        buttons = []
        for _, row in uploaded_df.iterrows():
            filename = row['Document Name']
            delete_button = create_delete_button(filename)
            buttons.append(HBox([delete_button]))
        display(uploaded_df)
        display(VBox(buttons))

# Update the table with buttons
update_table_with_buttons()


# In[10]:


# # Create and display the app layout
# app_layout = VBox([
#     widgets.HTML("<h3>Uploaded Documents</h3>"),
#     output,
#     widgets.HTML("<h4>Upload New Files</h4>"),
#     username_input,
#     upload_widget,
#     upload_button
# ])

# display(app_layout)


# In[ ]:





# In[11]:


from pypdf import PdfReader
from docx import Document
import re
from sklearn.feature_extraction.text import CountVectorizer


# In[12]:


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text_collection = [page.extract_text() for page in reader.pages]
    return "\n".join(text_collection)

# Function to extract text from Word
def extract_text_from_word(word_path):
    doc = Document(word_path)
    text_collection = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(text_collection)

# Function to clean resume text
def clean_text(text):
    text = re.sub('httpS+s*', ' ', text)  # Remove URLs
    text = re.sub('#S+', '', text)  # Remove hashtags
    text = re.sub('@S+', '  ', text)  # Remove mentions
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', text)  # Remove punctuations
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.strip()

# Function to extract candidate name from the file name
def extract_candidate_name(file_name):
    name_without_ext = os.path.splitext(file_name)[0]
    name_cleaned = re.sub(r'[_\-]+', ' ', name_without_ext)
    name_cleaned = re.sub(r'\b(resume|cv|profile|nov\d{4}|doc)\b', '', name_cleaned, flags=re.IGNORECASE)
    name_cleaned = ' '.join(word.capitalize() for word in name_cleaned.split())
    return name_cleaned

# Function to filter invalid n-grams
def is_valid_phrase(phrase):
    if re.search(r'\d', phrase):  # Contains any digit
        return False
    if re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|gpa)\b', phrase, re.IGNORECASE):
        return False  # Contains months or 'gpa'
    if len(phrase) < 3:  # Ignore very short phrases
        return False
    return True

# Function to extract top N keywords or phrases and their frequencies
def extract_key_skills_with_freq(text, n=10, ngram_range=(1, 3)):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english', min_df=1)
    X = vectorizer.fit_transform([text])
    freq = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
    sorted_freq = sorted(freq, key=lambda x: x[1], reverse=True)
    filtered_skills = {word: count for word, count in sorted_freq if is_valid_phrase(word)}
    top_n_skills = dict(list(filtered_skills.items())[:n])
    return top_n_skills

# Function to calculate keyword matches for each category
def categorize_resume(text, terms):
    category_scores = {category: 0 for category in terms.keys()}
    for category, keywords in terms.items():
        for keyword in keywords:
            matches = re.findall(r'\b' + re.escape(keyword) + r'\b', text.lower())
            category_scores[category] += len(matches)
    return category_scores

# Terms for categorization
terms = {
    'Quality/Six Sigma': ['black belt', 'capability analysis', 'control charts', 'doe', 'dmaic', 'fishbone'],
    'Operations Management': ['automation', 'bottleneck', 'constraints', 'cycle time', 'efficiency', 'fmea'],
    'Supply Chain': ['abc analysis', 'apics', 'customer', 'customs', 'delivery', 'distribution'],
    'Project Management': ['administration', 'agile', 'budget', 'cost', 'direction', 'feasibility analysis'],
    'Data Analytics': ['analytics', 'api', 'aws', 'big data', 'business intelligence', 'clustering'],
    'Healthcare': ['adverse events', 'care', 'clinic', 'cphq', 'ergonomics', 'healthcare'],
    'Cloud': ['aws', 'azure', 'gcp', 'cloud computing', 'cloud architecture', 'cloud deployment'],
    'Software Development': ['programming', 'coding', 'software engineering', 'software design', 'agile', 'scrum'],
    'Visualization Board': ['tableau', 'power bi', 'visualization', 'dashboards', 'data visualization'],
    'Process or Flow Automation': ['process mapping', 'process automation', 'workflow automation', 'rpa'],
    'Database': ['sql', 'nosql', 'mysql', 'postgresql', 'oracle', 'database design'],
    'Machine Learning and Modelling': ['machine learning', 'deep learning', 'predictive modeling', 'clustering']
}

# Function to process resumes
def process_resumes(directory_path, terms, n_skills=10, ngram_range=(2, 3)):
    global uploaded_df
    resume_data = []

    # Get the list of files in the directory
    files = [file for file in os.listdir(directory_path) if file.endswith(('.pdf', '.docx'))]
    
    # Sort files by name to ensure consistent ID assignment
    files.sort()

    for idx, file_name in enumerate(files):
        file_path = os.path.join(directory_path, file_name)
        
        if file_name.endswith('.pdf'):
            resume_text = extract_text_from_pdf(file_path)
        elif file_name.endswith('.docx'):
            resume_text = extract_text_from_word(file_path)
        
        cleaned_text = clean_text(resume_text)
        candidate_name = extract_candidate_name(file_name)
        key_skills = extract_key_skills_with_freq(cleaned_text, n=n_skills, ngram_range=ngram_range)
        category_scores = categorize_resume(cleaned_text, terms)
        expertise_area = max(category_scores, key=category_scores.get)

        resume_data.append({
            "ID": idx + 1,  # Ensure ID starts from 1 and increments correctly
            "File Name": file_name,
            "Candidate Name": candidate_name,
            "Resume": resume_text,
            "cleaned_resume": cleaned_text,
            "key skills": key_skills,
            "Expertise Area": expertise_area,
            "Category Scores": category_scores
        })
    
    return pd.DataFrame(resume_data)

# Process button
process_button = Button(description="Process Files", button_style='info')

def on_process_click(b):
    global uploaded_df
    with output:
        clear_output()
        print("Processing files...")
        uploaded_df = process_resumes(directory_path, terms, n_skills=10, ngram_range=(2, 3))
        print("Uploaded files have been processed. Here's the extracted data:")
        display(uploaded_df)

process_button.on_click(on_process_click)


# In[13]:


# # Create and display the app layout
# app = VBox([
#     widgets.HTML("<h3>Process and Analyze Uploaded Resumes</h3>"),
#     process_button,
#     output
# ])

# display(app)


# In[ ]:





# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


# 可视化函数
def visualize_data(df, id_value=None, name_value=None, visualization_type='Key Skills'):
    if id_value is not None:
        row = df[df['ID'] == id_value]
    elif name_value is not None:
        row = df[df['Candidate Name'].str.lower() == name_value.lower()]
    else:
        print("Please provide either an ID or a Candidate Name.")
        return
    
    if row.empty:
        if id_value is not None:
            print(f"No record found for ID {id_value}")
        else:
            print(f"No record found for Candidate Name '{name_value}'")
        return
    
    candidate_name = row['Candidate Name'].iloc[0]
    
    if visualization_type == 'Key Skills':
        key_skills = row['key skills'].iloc[0]
        
        # Bar chart for key skills
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(key_skills.values()), y=list(key_skills.keys()), palette="viridis")
        plt.title(f"Key Skills for {candidate_name} (Bar Chart)", fontsize=16)
        plt.xlabel("Frequency", fontsize=12)
        plt.ylabel("Skills", fontsize=12)
        plt.show()
        
        # Pie chart for key skills
        plt.figure(figsize=(8, 8))
        plt.pie(
            key_skills.values(), 
            labels=key_skills.keys(), 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=sns.color_palette('viridis', len(key_skills))
        )
        plt.title(f"Key Skills for {candidate_name} (Pie Chart)", fontsize=16)
        plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
        plt.show()
    
    elif visualization_type == 'Category Scores':
        category_scores = row['Category Scores'].iloc[0]
        
        # Bar chart for category scores
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(category_scores.values()), y=list(category_scores.keys()), palette="magma")
        plt.title(f"Category Scores for {candidate_name} (Bar Chart)", fontsize=16)
        plt.xlabel("Scores", fontsize=12)
        plt.ylabel("Categories", fontsize=12)
        plt.show()
        
        # Pie chart for category scores
        plt.figure(figsize=(8, 8))
        plt.pie(
            category_scores.values(), 
            labels=category_scores.keys(), 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=sns.color_palette('magma', len(category_scores))
        )
        plt.title(f"Category Scores for {candidate_name} (Pie Chart)", fontsize=16)
        plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
        plt.show()
    else:
        print(f"Visualization type '{visualization_type}' is not supported.")

# 可视化输入组件
id_input = widgets.IntText(description="Enter ID:")
name_input = widgets.Text(description="Enter Name:")
visualization_dropdown = widgets.Dropdown(
    options=['Key Skills', 'Category Scores'],
    description="Visualize:",
    value='Key Skills'
)

# 可视化按钮
visualization_button = widgets.Button(description="Visualize", button_style='warning')

# 可视化按钮点击事件
def on_visualize_click(b):
    clear_output(wait=True)
    display(id_input, name_input, visualization_dropdown, visualization_button)
    
    id_value = id_input.value if id_input.value else None
    name_value = name_input.value if name_input.value else None
    visualization_type = visualization_dropdown.value
    
    visualize_data(uploaded_df, id_value=id_value, name_value=name_value, visualization_type=visualization_type)

visualization_button.on_click(on_visualize_click)


# In[16]:


visualization_tab = VBox([
    widgets.HTML("<h3>Query Candidate Skills by ID or Name</h3>"),
    id_input,
    name_input,
    visualization_dropdown,
    visualization_button
])


# In[ ]:





# In[17]:


# Function to count the frequency of a skill in the cleaned_resume column
def count_skill_frequency(df, skill):
    skill_lower = skill.lower()
    df[f"Frequency of '{skill}'"] = df['cleaned_resume'].apply(
        lambda x: len(re.findall(r'\b' + re.escape(skill_lower) + r'\b', x.lower()))
    )
    return df

# Function to get frequency columns
def get_frequency_columns(df):
    return [col for col in df.columns if col.startswith("Frequency of ")]

# Function to sort and display DataFrame by selected skill column
def sort_and_display(df, selected_skill_column):
    if selected_skill_column in df.columns:
        sorted_df = df.sort_values(selected_skill_column, ascending=False)
        display(sorted_df)
    else:
        print(f"No column found for: {selected_skill_column}")

# Initialize dropdown (initially disabled)
dropdown = widgets.Dropdown(description="Select Skill:", disabled=True)

# Create input box and button for user interaction
skill_input = widgets.Text(description="Enter Skill:", button_style='primary')
skill_button = widgets.Button(description="Count Skill", button_style='warning')

# Button click event handler for Count Skill
def on_skill_click(b):
    clear_output(wait=True)
    display(skill_input, skill_button, dropdown, sort_button)
    
    # Get user input skill
    skill = skill_input.value.strip()
    if not skill:
        print("Please enter a skill!")
        return
    
    # Count frequency of the skill
    updated_df = count_skill_frequency(uploaded_df, skill)
    display(updated_df)
    
    # Update dropdown options
    frequency_columns = get_frequency_columns(updated_df)
    if frequency_columns:
        dropdown.options = frequency_columns
        dropdown.disabled = False
    else:
        print("No frequency columns found in the DataFrame!")
        dropdown.disabled = True

# Bind button click event
skill_button.on_click(on_skill_click)

# Button for sorting by skill
sort_button = widgets.Button(description="Sort by Skill", button_style='success')

# Button click event handler for Sort by Skill
def on_sort_click(b):
    clear_output(wait=True)
    display(skill_input, skill_button, dropdown, sort_button)
    
    if dropdown.disabled:
        print("No skill frequency columns found! Please count a skill first.")
        return
    
    selected_skill_column = dropdown.value
    sort_and_display(uploaded_df, selected_skill_column)

# Bind button click event
sort_button.on_click(on_sort_click)

# Display the initial widgets
#display(VBox([skill_input, skill_button, dropdown, sort_button]))


# In[ ]:





# In[18]:


# Function to calculate weighted scores
def calculate_weighted_scores(df, skill_weights):
    df['Weighted Score'] = df.apply(
        lambda row: sum(row[skill] * (weight / 100) for skill, weight in skill_weights.items()),
        axis=1
    )
    
    # Move 'Weighted Score' column to the end of all 'Frequency of xxx' columns
    skill_columns = get_frequency_columns(df)
    if skill_columns:
        last_frequency_index = df.columns.get_loc(skill_columns[-1]) + 1
        df.insert(last_frequency_index, 'Weighted Score', df.pop('Weighted Score'))
    
    # Sort the DataFrame by 'Weighted Score' in descending order
    df = df.sort_values(by='Weighted Score', ascending=False)
    
    return df

# Function to display the form for skill weights
def display_skill_weight_form(df):
    skill_columns = get_frequency_columns(df)
    if not skill_columns:
        print("No skill frequency columns found!")
        return

    # Create input boxes for each skill weight
    skill_weight_inputs = {}
    for skill in skill_columns:
        skill_name = skill.replace("Frequency of ", "").strip("'")
        skill_weight_inputs[skill] = widgets.FloatText(
            description=f"{skill_name} Weight (%):",
            min=0,
            max=100,
            step=1
        )

    # Create a submit button
    submit_button = widgets.Button(description="Calculate Weighted Score", button_style='success')

    # Button click event handler
    def on_submit_click(b):
        skill_weights = {skill: input_box.value for skill, input_box in skill_weight_inputs.items()}
        total_weight = sum(skill_weights.values())
        
        if total_weight != 100:
            print(f"Total weight must sum to 100%, but got {total_weight:.2f}%. Please try again.")
            return
        
        updated_df = calculate_weighted_scores(df, skill_weights)
        print("Weighted scores calculated successfully!")
        display(updated_df)

    submit_button.on_click(on_submit_click)

    # Display the form
    form_items = [VBox([input_box]) for input_box in skill_weight_inputs.values()]
    display(VBox(form_items + [submit_button], layout=widgets.Layout(margin="20px 0px")))

# Button to trigger the skill weight form
weighted_score_button = widgets.Button(description="Calculate Weighted Score", button_style='info')

def on_weighted_score_click(b):
    with output:
        clear_output()
        display_skill_weight_form(uploaded_df)

weighted_score_button.on_click(on_weighted_score_click)


# In[19]:


# # Create and display the app layout
# app = VBox([
#     widgets.HTML("<h3>Calculate Weighted Score for Candidates</h3>"),
#     weighted_score_button,
#     output
# ])

# display(app)


# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import seaborn as sns

# 初始化全局变量
tab = None
tab_initialized = False  # 标记 tab 是否已经初始化

# 为每个模块创建独立的 Output 组件
file_upload_output = widgets.Output()
file_processing_output = widgets.Output()
skill_query_output = widgets.Output()
weighted_score_output = widgets.Output()
visualization_output = widgets.Output()

# 定义选项卡界面
def create_tab_interface():
    global tab, tab_initialized
    
    # 如果 tab 已经初始化，则直接返回
    if tab_initialized:
        return
    
    # 文件上传与查看模块
    file_upload_tab = VBox([
        widgets.HTML("<h3>Uploaded Documents</h3>"),
        file_upload_output,
        widgets.HTML("<h4>Upload New Files</h4>"),
        username_input,
        upload_widget,
        upload_button
    ])

    # 文件处理与技能提取模块
    file_processing_tab = VBox([
        widgets.HTML("<h3>Process and Analyze Uploaded Resumes</h3>"),
        file_processing_output,
        process_button
    ])

    # 技能查询与排序模块
    skill_query_tab = VBox([
        widgets.HTML("<h3>Skill Query and Sorting</h3>"),
        skill_query_output,
        skill_input,
        skill_button,
        dropdown,
        sort_button
    ])

    # 加权分数计算模块
    weighted_score_tab = VBox([
        widgets.HTML("<h3>Calculate Weighted Score for Candidates</h3>"),
        weighted_score_output,
        weighted_score_button
    ])

    # 可视化模块
    visualization_tab = VBox([
        widgets.HTML("<h3>Query Candidate Skills by ID or Name</h3>"),
        visualization_output,
        id_input,
        name_input,
        visualization_dropdown,
        visualization_button
    ])

    # 创建选项卡
    tab = widgets.Tab()
    tab.children = [file_upload_tab, file_processing_tab, skill_query_tab, weighted_score_tab, visualization_tab]
    tab.titles = ["File Upload", "File Processing", "Skill Query", "Weighted Score", "Visualization"]
    
    # 标记 tab 已初始化
    tab_initialized = True

    # 显示选项卡
    display(tab)

# 初始化选项卡界面（仅运行一次）
create_tab_interface()

# 文件上传按钮点击事件
def on_upload_click(b):
    with file_upload_output:
        clear_output(wait=True)
        # 文件上传逻辑
        print("Files uploaded successfully!")
        display(uploaded_df)

upload_button.on_click(on_upload_click)

# 文件处理按钮点击事件
def on_process_click(b):
    with file_processing_output:
        clear_output(wait=True)
        # 文件处理逻辑
        print("Processing files...")
        uploaded_df = process_resumes(directory_path, terms)
        display(uploaded_df)

process_button.on_click(on_process_click)

# 技能查询按钮点击事件
def on_skill_click(b):
    with skill_query_output:
        clear_output(wait=True)
        # 技能查询逻辑
        skill = skill_input.value.strip()
        if skill:
            updated_df = count_skill_frequency(uploaded_df, skill)
            display(updated_df)
        else:
            print("Please enter a skill!")

skill_button.on_click(on_skill_click)

# 加权分数计算按钮点击事件
def on_weighted_score_click(b):
    with weighted_score_output:
        clear_output(wait=True)
        # 加权分数计算逻辑
        display_skill_weight_form(uploaded_df)

weighted_score_button.on_click(on_weighted_score_click)

# 可视化按钮点击事件
def on_visualize_click(b):
    with visualization_output:
        clear_output(wait=True)
        # 可视化逻辑
        id_value = id_input.value if id_input.value else None
        name_value = name_input.value if name_input.value else None
        visualization_type = visualization_dropdown.value
        visualize_data(uploaded_df, id_value=id_value, name_value=name_value, visualization_type=visualization_type)

visualization_button.on_click(on_visualize_click)


# In[ ]:





# In[ ]:





# In[23]:


#pip install streamlit pandas matplotlib seaborn


# In[24]:


import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pypdf import PdfReader
from docx import Document
import re
from sklearn.feature_extraction.text import CountVectorizer

# 初始化全局变量
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = pd.DataFrame(columns=['Document Name', 'Uploaded By'])

# 文件上传与查看模块
def file_upload_module():
    st.header("File Upload and View")
    
    # 文件上传
    uploaded_files = st.file_uploader("Upload files (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    username = st.text_input("Uploaded By:", placeholder="Enter your name")
    
    if st.button("Upload Files"):
        if not username.strip():
            st.error("Please enter your name before uploading!")
            return
        
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_path = os.path.join(os.getcwd(), file_name)
            
            # 检查文件是否已存在
            if file_name in st.session_state.uploaded_df['Document Name'].values:
                st.error(f"File '{file_name}' already exists! Please upload a different file.")
                return
            
            # 保存文件
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                new_row = {'Document Name': file_name, 'Uploaded By': username.strip()}
                st.session_state.uploaded_df = pd.concat([st.session_state.uploaded_df, pd.DataFrame([new_row])], ignore_index=True)
            except Exception as e:
                st.error(f"Failed to upload file '{file_name}': {str(e)}")
                return
        
        st.success("Files uploaded successfully!")
    
    # 显示已上传的文件列表
    st.subheader("Uploaded Documents")
    st.dataframe(st.session_state.uploaded_df)

# 文件处理与技能提取模块
def file_processing_module():
    st.header("File Processing and Skill Extraction")
    
    if st.button("Process Files"):
        st.session_state.processed_df = process_resumes(os.getcwd(), terms)
        st.success("Files processed successfully!")
    
    if 'processed_df' in st.session_state:
        st.subheader("Processed Data")
        st.dataframe(st.session_state.processed_df)

# 技能查询与排序模块
def skill_query_module():
    st.header("Skill Query and Sorting")
    
    skill = st.text_input("Enter Skill:", placeholder="e.g., Python")
    if st.button("Count Skill"):
        if skill.strip():
            updated_df = count_skill_frequency(st.session_state.processed_df, skill)
            st.subheader("Skill Frequency")
            st.dataframe(updated_df)
        else:
            st.error("Please enter a skill!")
    
    if st.button("Sort by Skill"):
        if skill.strip():
            sorted_df = st.session_state.processed_df.sort_values(by=f"Frequency of '{skill}'", ascending=False)
            st.subheader("Sorted Data")
            st.dataframe(sorted_df)
        else:
            st.error("Please enter a skill!")

# 加权分数计算模块
def weighted_score_module():
    st.header("Weighted Score Calculation")
    
    if 'processed_df' in st.session_state:
        skill_columns = [col for col in st.session_state.processed_df.columns if col.startswith("Frequency of ")]
        if skill_columns:
            st.subheader("Set Skill Weights")
            skill_weights = {}
            for skill in skill_columns:
                skill_name = skill.replace("Frequency of ", "").strip("'")
                skill_weights[skill] = st.slider(f"{skill_name} Weight (%)", 0, 100, 0)
            
            if st.button("Calculate Weighted Score"):
                total_weight = sum(skill_weights.values())
                if total_weight != 100:
                    st.error(f"Total weight must sum to 100%, but got {total_weight}%. Please try again.")
                else:
                    updated_df = calculate_weighted_scores(st.session_state.processed_df, skill_weights)
                    st.subheader("Weighted Scores")
                    st.dataframe(updated_df)
        else:
            st.error("No skill frequency columns found! Please process files first.")
    else:
        st.error("No processed data found! Please process files first.")

# 可视化模块
def visualization_module():
    st.header("Visualization")
    
    if 'processed_df' in st.session_state:
        id_value = st.number_input("Enter ID:", min_value=1, step=1)
        name_value = st.text_input("Enter Name:", placeholder="e.g., John Doe")
        visualization_type = st.selectbox("Visualize:", ["Key Skills", "Category Scores"])
        
        if st.button("Visualize"):
            visualize_data(st.session_state.processed_df, id_value=id_value, name_value=name_value, visualization_type=visualization_type)
    else:
        st.error("No processed data found! Please process files first.")

# 主界面
def main():
    st.sidebar.title("Navigation")
    modules = {
        "File Upload": file_upload_module,
        "File Processing": file_processing_module,
        "Skill Query": skill_query_module,
        "Weighted Score": weighted_score_module,
        "Visualization": visualization_module
    }
    selected_module = st.sidebar.radio("Go to", list(modules.keys()))
    
    # 显示选中的模块
    modules[selected_module]()

# 运行 Streamlit 应用
if __name__ == "__main__":
    main()






