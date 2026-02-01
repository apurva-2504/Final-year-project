from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify, session
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import plotly
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Create necessary folders
os.makedirs('uploads', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Global variables to store data
students_data = None
internships_data = None
match_matrix = None
allocations = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Feature engineering functions
def parse_skills_list(skills_str):
    if pd.isna(skills_str):
        return []
    return [skill.strip() for skill in str(skills_str).split(',')]

def calculate_skill_match(student_skills, required_skills, preferred_skills=None):
    student_skills_set = set(parse_skills_list(student_skills))
    required_skills_set = set(parse_skills_list(required_skills))
    
    if len(required_skills_set) == 0:
        return 0.0
    
    required_match = len(student_skills_set.intersection(required_skills_set)) / len(required_skills_set)
    
    if preferred_skills:
        preferred_skills_set = set(parse_skills_list(preferred_skills))
        if len(preferred_skills_set) > 0:
            preferred_match = len(student_skills_set.intersection(preferred_skills_set)) / len(preferred_skills_set)
            return 0.8 * required_match + 0.2 * preferred_match
    
    return required_match

def calculate_interest_match(student_interests, internship_domain):
    student_interests_list = [interest.strip().lower() for interest in parse_skills_list(student_interests)]
    domain_lower = str(internship_domain).lower()
    
    for interest in student_interests_list:
        if interest in domain_lower or domain_lower in interest:
            return 1.0
    
    keyword_matches = {
        'web': ['web development', 'frontend', 'backend', 'full stack'],
        'data': ['data science', 'data analysis', 'machine learning', 'ai'],
        'mobile': ['mobile development', 'android', 'ios'],
        'cloud': ['cloud computing', 'devops', 'aws'],
        'security': ['cybersecurity', 'information security']
    }
    
    for key, related in keyword_matches.items():
        if key in domain_lower:
            for interest in student_interests_list:
                if any(rel in interest for rel in related):
                    return 0.7
    
    return 0.0

def calculate_cgpa_score(student_cgpa, min_cgpa_required):
    if student_cgpa < min_cgpa_required:
        return 0.0
    
    excess = student_cgpa - min_cgpa_required
    max_possible_excess = 10.0 - min_cgpa_required
    
    if max_possible_excess == 0:
        return 1.0
    
    return min(1.0, 0.5 + (excess / max_possible_excess) * 0.5)

def calculate_location_match(student_locations, internship_location):
    student_locations_list = [loc.strip() for loc in parse_skills_list(student_locations)]
    return 1.0 if internship_location in student_locations_list else 0.0

def calculate_branch_match(student_branch, preferred_branches):
    preferred_branches_list = [branch.strip() for branch in parse_skills_list(preferred_branches)]
    
    if student_branch in preferred_branches_list:
        return 1.0
    
    cs_related = ['Computer Science', 'Information Technology']
    if student_branch in cs_related and any(branch in cs_related for branch in preferred_branches_list):
        return 0.8
    
    return 0.0

def calculate_experience_match(student_experience, experience_required):
    if experience_required == 'Not Required':
        return 1.0
    return 1.0 if student_experience == 'Yes' else 0.5

# Routes
@app.route('/')
def index():
    print("Index route accessed!")  # Debug line
    print(f"Template folder: {app.template_folder}")  # Debug line
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global students_data, internships_data
    
    if request.method == 'POST':
        # Check if files are present
        if 'students_file' not in request.files or 'internships_file' not in request.files:
            flash('Please upload both files', 'danger')
            return redirect(request.url)
        
        students_file = request.files['students_file']
        internships_file = request.files['internships_file']
        
        # Check if files are selected
        if students_file.filename == '' or internships_file.filename == '':
            flash('Please select both files', 'danger')
            return redirect(request.url)
        
        # Validate and save files
        if students_file and allowed_file(students_file.filename) and internships_file and allowed_file(internships_file.filename):
            students_filename = secure_filename(students_file.filename)
            internships_filename = secure_filename(internships_file.filename)
            
            students_path = os.path.join(app.config['UPLOAD_FOLDER'], 'students_data.csv')
            internships_path = os.path.join(app.config['UPLOAD_FOLDER'], 'internships_data.csv')
            
            students_file.save(students_path)
            internships_file.save(internships_path)
            
            try:
                students_data = pd.read_csv(students_path)
                internships_data = pd.read_csv(internships_path)
                
                session['students_count'] = len(students_data)
                session['internships_count'] = len(internships_data)
                session['total_positions'] = int(internships_data['positions_available'].sum())
                
                flash(f'Files uploaded successfully! {len(students_data)} students and {len(internships_data)} internships loaded.', 'success')
                return redirect(url_for('data_preview'))
            
            except Exception as e:
                flash(f'Error reading files: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('Only CSV files are allowed', 'danger')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/data_preview')
def data_preview():
    global students_data, internships_data
    
    if students_data is None or internships_data is None:
        flash('Please upload data files first', 'warning')
        return redirect(url_for('upload'))
    
    students_preview = students_data.head(10).to_html(classes='table table-striped table-hover', index=False)
    internships_preview = internships_data.head(10).to_html(classes='table table-striped table-hover', index=False)
    
    stats = {
        'students_count': len(students_data),
        'internships_count': len(internships_data),
        'total_positions': int(internships_data['positions_available'].sum()),
        'avg_cgpa': round(students_data['cgpa'].mean(), 2),
        'avg_stipend': int(internships_data['stipend'].mean())
    }
    
    return render_template('data_preview.html', 
                         students_preview=students_preview, 
                         internships_preview=internships_preview,
                         stats=stats)

@app.route('/configure', methods=['GET', 'POST'])
def configure():
    if request.method == 'POST':
        # Get weights from form
        weights = {
            'skill_match': float(request.form.get('skill_weight', 0.40)),
            'interest_match': float(request.form.get('interest_weight', 0.20)),
            'cgpa_match': float(request.form.get('cgpa_weight', 0.15)),
            'location_match': float(request.form.get('location_weight', 0.15)),
            'branch_match': float(request.form.get('branch_weight', 0.05)),
            'experience_match': float(request.form.get('experience_weight', 0.05))
        }
        
        min_threshold = float(request.form.get('min_threshold', 0.30))
        
        # Store in session
        session['weights'] = weights
        session['min_threshold'] = min_threshold
        
        flash('Configuration saved successfully!', 'success')
        return redirect(url_for('process'))
    
    # Default weights
    default_weights = {
        'skill_match': 0.40,
        'interest_match': 0.20,
        'cgpa_match': 0.15,
        'location_match': 0.15,
        'branch_match': 0.05,
        'experience_match': 0.05
    }
    
    return render_template('configure.html', weights=default_weights, min_threshold=0.30)

@app.route('/process')
def process():
    global students_data, internships_data, match_matrix
    
    if students_data is None or internships_data is None:
        flash('Please upload data files first', 'warning')
        return redirect(url_for('upload'))
    
    # Get weights from session or use defaults
    weights = session.get('weights', {
        'skill_match': 0.40,
        'interest_match': 0.20,
        'cgpa_match': 0.15,
        'location_match': 0.15,
        'branch_match': 0.05,
        'experience_match': 0.05
    })
    
    # Generate match matrix
    match_data = []
    
    for _, student in students_data.iterrows():
        for _, internship in internships_data.iterrows():
            skill_score = calculate_skill_match(
                student['technical_skills'], 
                internship['required_skills'],
                internship['preferred_skills']
            )
            
            interest_score = calculate_interest_match(
                student['interests'],
                internship['domain']
            )
            
            cgpa_score = calculate_cgpa_score(
                student['cgpa'],
                internship['min_cgpa']
            )
            
            location_score = calculate_location_match(
                student['location_preferences'],
                internship['location']
            )
            
            branch_score = calculate_branch_match(
                student['branch'],
                internship['preferred_branches']
            )
            
            experience_score = calculate_experience_match(
                student['previous_experience'],
                internship['experience_required']
            )
            
            total_score = (
                weights['skill_match'] * skill_score +
                weights['interest_match'] * interest_score +
                weights['cgpa_match'] * cgpa_score +
                weights['location_match'] * location_score +
                weights['branch_match'] * branch_score +
                weights['experience_match'] * experience_score
            )
            
            match_data.append({
                'student_id': student['student_id'],
                'student_name': student['name'],
                'internship_id': internship['internship_id'],
                'company_name': internship['company_name'],
                'role': internship['role'],
                'skill_match': skill_score,
                'interest_match': interest_score,
                'cgpa_match': cgpa_score,
                'location_match': location_score,
                'branch_match': branch_score,
                'experience_match': experience_score,
                'total_match_score': total_score
            })
    
    match_matrix = pd.DataFrame(match_data)
    match_matrix.to_csv('results/match_matrix.csv', index=False)
    
    session['match_matrix_generated'] = True
    flash(f'Match matrix generated successfully! {len(match_matrix):,} combinations evaluated.', 'success')
    
    return redirect(url_for('allocate'))

@app.route('/allocate')
def allocate():
    global students_data, internships_data, match_matrix, allocations
    
    if match_matrix is None:
        flash('Please generate match matrix first', 'warning')
        return redirect(url_for('configure'))
    
    min_threshold = session.get('min_threshold', 0.30)
    
    # Filter eligible matches
    eligible_matches = match_matrix[match_matrix['total_match_score'] >= min_threshold].copy()
    eligible_matches = eligible_matches.sort_values('total_match_score', ascending=False).reset_index(drop=True)
    
    # Initialize tracking
    allocated_students = set()
    remaining_positions = {}
    for _, internship in internships_data.iterrows():
        remaining_positions[internship['internship_id']] = internship['positions_available']
    
    # Run greedy allocation
    allocation_list = []
    for _, match in eligible_matches.iterrows():
        student_id = match['student_id']
        internship_id = match['internship_id']
        
        if student_id in allocated_students:
            continue
        
        if remaining_positions.get(internship_id, 0) <= 0:
            continue
        
        allocation_list.append(match.to_dict())
        allocated_students.add(student_id)
        remaining_positions[internship_id] -= 1
    
    allocations = pd.DataFrame(allocation_list)
    allocations.to_csv('results/final_allocations.csv', index=False)
    
    # Get unallocated students
    unallocated = students_data[~students_data['student_id'].isin(allocated_students)]
    unallocated.to_csv('results/unallocated_students.csv', index=False)
    
    session['allocations_count'] = len(allocations)
    session['unallocated_count'] = len(unallocated)
    session['allocation_rate'] = round(len(allocations) / len(students_data) * 100, 2)
    
    flash(f'Allocation complete! {len(allocations)} students allocated.', 'success')
    return redirect(url_for('results'))

@app.route('/results')
def results():
    global allocations, students_data, internships_data
    
    if allocations is None:
        flash('Please run allocation first', 'warning')
        return redirect(url_for('configure'))
    
    stats = {
        'total_students': len(students_data),
        'total_internships': len(internships_data),
        'total_positions': int(internships_data['positions_available'].sum()),
        'allocated': len(allocations),
        'unallocated': len(students_data) - len(allocations),
        'allocation_rate': round(len(allocations) / len(students_data) * 100, 2),
        'avg_match_score': round(allocations['total_match_score'].mean(), 4),
        'min_match_score': round(allocations['total_match_score'].min(), 4),
        'max_match_score': round(allocations['total_match_score'].max(), 4)
    }
    
    # Top companies
    top_companies = allocations['company_name'].value_counts().head(10).to_dict()
    
    # Score distribution
    score_ranges = [
        (0.9, 1.0, "Excellent"),
        (0.8, 0.9, "Very Good"),
        (0.7, 0.8, "Good"),
        (0.6, 0.7, "Fair"),
        (0.0, 0.6, "Acceptable")
    ]
    
    score_distribution = {}
    for min_score, max_score, label in score_ranges:
        count = len(allocations[(allocations['total_match_score'] >= min_score) & 
                                (allocations['total_match_score'] < max_score)])
        score_distribution[label] = count
    
    return render_template('results.html', 
                         stats=stats, 
                         top_companies=top_companies,
                         score_distribution=score_distribution)

@app.route('/visualizations')
def visualizations():
    global allocations
    
    if allocations is None:
        flash('Please run allocation first', 'warning')
        return redirect(url_for('configure'))
    
    # Create visualizations
    # 1. Match Score Distribution
    fig1 = px.histogram(allocations, x='total_match_score', nbins=30,
                       title='Match Score Distribution',
                       labels={'total_match_score': 'Match Score', 'count': 'Frequency'})
    fig1.update_layout(showlegend=False)
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 2. Top 10 Companies
    top_companies = allocations['company_name'].value_counts().head(10)
    fig2 = px.bar(x=top_companies.values, y=top_companies.index, orientation='h',
                 title='Top 10 Companies by Allocations',
                 labels={'x': 'Number of Students', 'y': 'Company'})
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 3. Feature Contributions
    feature_cols = ['skill_match', 'interest_match', 'cgpa_match', 'location_match', 'branch_match', 'experience_match']
    feature_means = allocations[feature_cols].mean()
    fig3 = px.bar(x=feature_cols, y=feature_means.values,
                 title='Average Feature Scores',
                 labels={'x': 'Feature', 'y': 'Average Score'})
    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 4. Score Quality Pie Chart
    score_ranges = [
        (0.9, 1.0, "Excellent"),
        (0.8, 0.9, "Very Good"),
        (0.7, 0.8, "Good"),
        (0.6, 0.7, "Fair"),
        (0.0, 0.6, "Acceptable")
    ]
    
    labels = []
    values = []
    for min_score, max_score, label in score_ranges:
        count = len(allocations[(allocations['total_match_score'] >= min_score) & 
                                (allocations['total_match_score'] < max_score)])
        labels.append(label)
        values.append(count)
    
    fig4 = px.pie(values=values, names=labels, title='Allocation Quality Distribution')
    graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('visualizations.html',
                         graph1JSON=graph1JSON,
                         graph2JSON=graph2JSON,
                         graph3JSON=graph3JSON,
                         graph4JSON=graph4JSON)

@app.route('/view_allocations')
def view_allocations():
    global allocations
    
    if allocations is None:
        flash('Please run allocation first', 'warning')
        return redirect(url_for('configure'))
    
    page = request.args.get('page', 1, type=int)
    per_page = 50
    
    total = len(allocations)
    start = (page - 1) * per_page
    end = start + per_page
    
    paginated_data = allocations.iloc[start:end]
    allocations_html = paginated_data.to_html(classes='table table-striped table-hover', index=False)
    
    total_pages = (total + per_page - 1) // per_page
    
    return render_template('view_allocations.html',
                         allocations_html=allocations_html,
                         page=page,
                         total_pages=total_pages,
                         total=total)

@app.route('/search_student', methods=['GET', 'POST'])
def search_student():
    global allocations, students_data
    
    if request.method == 'POST':
        search_query = request.form.get('search_query', '').strip()
        
        if allocations is None:
            flash('Please run allocation first', 'warning')
            return redirect(url_for('configure'))
        
        # Search in allocations
        result = allocations[
            (allocations['student_id'].str.contains(search_query, case=False, na=False)) |
            (allocations['student_name'].str.contains(search_query, case=False, na=False))
        ]
        
        if len(result) > 0:
            student_info = result.iloc[0].to_dict()
            return render_template('search_student.html', student_info=student_info, found=True)
        else:
            # Check if student exists but not allocated
            student_exists = students_data[
                (students_data['student_id'].str.contains(search_query, case=False, na=False)) |
                (students_data['name'].str.contains(search_query, case=False, na=False))
            ]
            
            if len(student_exists) > 0:
                flash(f'Student found but not allocated: {student_exists.iloc[0]["name"]}', 'warning')
            else:
                flash('Student not found', 'danger')
            
            return render_template('search_student.html', found=False)
    
    return render_template('search_student.html', found=None)

@app.route('/download/<file_type>')
def download(file_type):
    try:
        if file_type == 'allocations':
            return send_file('results/final_allocations.csv', as_attachment=True)
        elif file_type == 'unallocated':
            return send_file('results/unallocated_students.csv', as_attachment=True)
        elif file_type == 'match_matrix':
            return send_file('results/match_matrix.csv', as_attachment=True)
        else:
            flash('Invalid file type', 'danger')
            return redirect(url_for('results'))
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'danger')
        return redirect(url_for('results'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True)