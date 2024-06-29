from flask import Flask,request,render_template
import pickle
from PyPDF2 import PdfReader
import re
app = Flask(__name__)


with open('models/rf_classifier_categorization.pkl', 'rb') as f:
    rf_classifier_categorization = pickle.load(f)
with open('models/tfidf_vectorizer_categorization.pkl', 'rb') as f:
    tfidf_vectorizer_categorization = pickle.load(f)
with open('models/rf_classifier_job_recommendation.pkl', 'rb') as f:
    rf_classifier_job_recommendation = pickle.load(f)
with open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb') as f:
    tfidf_vectorizer_job_recommendation = pickle.load(f)

def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

def cleanResume(txt):
    txt = re.sub('http\S+\s', ' ', txt)
    txt = re.sub('RT|cc', ' ', txt)
    txt = re.sub('#\S+\s', ' ', txt)
    txt = re.sub('@\S+', '  ', txt)
    txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub('\s+', ' ', txt)
    return txt

def predict_category(resume_text):
    resume_text= cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
    return predicted_category

# Prediction and Category Name
def job_recommendation(resume_text):
    resume_text= cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    predicted_category = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return predicted_category

#Resume Parsing
def extract_name_from_resume(text):
    match = re.search(r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)", text)
    return match.group() if match else None

def extract_email_from_resume(text):
    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    return match.group() if match else None

def extract_contact_number_from_resume(text):
    match = re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text)
    return match.group() if match else None

def extract_skills_from_resume(text):
    skills_list = [
        "3D Modeling", "Adobe XD", "Agile", "After Effects", "Analytics", "Analytical Skills", "Android Development",
        "Angular", "Animation", "Ansible", "AppDynamics", "Application Administration", "Application Security",
        "Application Support", "ASP.NET", "Attention to Detail", "AutoCAD", "AWS", "Axure", "Azure", "Azure DevOps",
        "Backbone.js", "Basecamp", "Big Data", "BigPanda", "Bitbucket", "Blockchain", "Blender", "Blogging",
        "BlueJeans", "Bootstrap", "Box", "Branding", "Broadcasting", "Business Administration", "Business Analysis",
        "Business Continuity", "Business Development", "Business Intelligence", "Business Partnering",
        "Business Support",
        "C", "C++", "CakePHP", "Capacity Planning", "Case Management", "Cassandra", "Change Management", "Chef",
        "Cherwell", "Child Welfare", "Cinema 4D", "Citrix", "Client Relations", "Client Support",
        "Cloud Administration",
        "Cloud Computing", "Cloud Security", "Cloud Support", "CodeIgniter", "Collaboration", "Compliance",
        "Compliance Administration", "Content Creation", "Content Writing", "Contract Management",
        "Corporate Social Responsibility",
        "Corrections", "Cost Management", "Creativity", "Critical Thinking", "CSS", "Curriculum Development",
        "Customer Relationship Management", "Customer Service", "Customer Support", "Cyber Threat Intelligence",
        "Cybersecurity", "Data Administration", "Data Analysis", "Data Entry", "Data Mining", "Data Science",
        "Data Support",
        "Data Visualization", "Database Administration", "Database Support", "Datadog", "Dataiku", "Decision Making",
        "Deep Learning", "Desktop Support", "DevOps", "DevOps Support", "Digital Media", "Disaster Recovery", "Docker",
        "Documentation", "Django", "DNS", "Docker", "Dropbox", "Dynatrace", "E-Learning", "Education",
        "Educational Technology", "Electronic Health Records", "ELK Stack", "Email Marketing", "Employee Engagement",
        "Employee Relations", "Engineering Administration", "Engineering Support", "Environmental Health", "ERP",
        "Event Management", "Event Planning", "Express.js", "Facebook Ads", "Facility Management", "Figma",
        "Financial Administration",
        "Financial Analysis", "Firewall", "First Aid", "Flask", "Flexibility", "Forensics", "Frontend Development",
        "Full Stack Development", "FusionCharts", "GCP", "ggplot2", "Gensim", "Ghostwriting", "Git", "GitHub",
        "GitHub Actions",
        "GitLab", "GitLab CI", "Go", "Google Ads", "Google Analytics", "Google Cloud", "Governance", "Grafana",
        "Graphic Design",
        "Group Policy", "Help Desk", "Help Desk Support", "Hibernate", "Highcharts", "HIPAA", "Hive", "HTML",
        "Human Resources",
        "IBM Cloud", "Identity Management", "Illustrator", "Image Processing", "Incident Handling", "Incident Response",
        "Information Security", "Information Technology", "Innovation", "Instructional Design", "InVision", "IPSec",
        "Java", "JavaScript", "Jenkins", "Jira", "Jira Service Desk", "jQuery", "JSON", "Kafka", "Kanban", "Kaseya",
        "Keras", "Kibana", "Kotlin", "Kubernetes", "LabTech", "Laravel", "Lean", "Lean Manufacturing",
        "Legislative Affairs",
        "Legal Administration", "Legal Compliance", "Lightroom", "Linux", "Load Balancing", "Logistics",
        "Logistics Administration",
        "Machine Learning", "Maintenance", "Maintenance Administration", "Manufacturing",
        "Manufacturing Administration",
        "MariaDB", "Marketing", "Marketing Administration", "Market Research", "Mastering", "MATLAB", "Medical Billing",
        "Medical Coding", "Medical Devices", "Mental Health", "Mental Health Counseling", "Mentoring", "Metasploit",
        "Microsoft Teams", "Microsoft Word", "Mobile Development", "MongoDB", "Moogsoft", "Motion Design", "MS Excel",
        "MS SQL",
        "MS Visio", "Multithreading", "Natural Language Processing", "Negotiation", "Network Administration",
        "Network Security", "Network Support", "New Relic", "Nginx", "Node.js", "Nursing", "OAuth2", "Objective-C",
        "Onboarding", "OpenShift", "OpenStack", "Oracle", "Organizational Development", "Origami", "Pandas",
        "Patient Advocacy",
        "Payroll", "PC Troubleshooting", "Perl", "PHP", "Pig", "PKI", "PMP", "Podcasting", "PostgreSQL", "Power BI",
        "Premiere Pro", "Presentation Skills", "Process Administration", "Process Improvement", "Procurement",
        "Procurement Administration", "Product Development", "Product Management", "Product Support", "Productivity",
        "Programming", "Project Administration", "Project Coordination", "Project Management", "Project Planning",
        "Project Support",
        "Public Health", "Public Relations", "Python", "PyTorch", "Quality Administration", "Quality Assurance",
        "Quality Control", "R", "Rancher", "React", "Reading", "Reasoning", "Red Team", "Redis",
        "Regulatory Administration",
        "Relational Databases", "Remote Support", "Research Administration", "Research Support", "Reverse Engineering",
        "Risk Management", "Ruby", "Ruby on Rails", "Rust", "SaaS", "Salesforce", "SaltStack", "SAP", "SAS", "Scala",
        "Scikit-learn",
        "Scrum", "Secure Coding", "Security Administration", "Security Architecture", "Security Engineering",
        "Security Operations", "SEO", "Server Administration", "Service Administration", "ServiceNow",
        "Service Support",
        "Shell Scripting", "Single Sign-On", "Six Sigma", "Slack", "Social Media", "Software Architecture",
        "Software Development", "Software Engineering", "Software Support", "SolarWinds", "Solidity",
        "Solution Architecture",
        "Spacy", "Spark", "Splunk", "Spring", "SPSS", "SQL", "Stakeholder Management", "Stata", "Statistical Analysis",
        "Stock Management", "Strategic Planning", "Stripe", "Swift", "Synchronous Programming", "System Administration",
        "Systems Engineering", "Tableau", "Technical Support", "Terraform", "TensorFlow", "Testing Administration",
        "ThousandEyes",
        "Three.js", "Time Management", "Trello", "Tribal Knowledge", "Troubleshooting", "TypeScript", "Unreal Engine",
        "User Experience", "User Interface", "UX Design", "Vagrant", "Version Control", "Video Editing",
        "Visual Design",
        "VPN", "Vue.js", "Web Design", "Web Development", "Webex", "WebGL", "Webpack", "Wireshark", "WordPress",
        "Xamarin", "XML", "Zeplin", "Zbrush", "Zendesk", "Zoom"
    ]
    skills= []
    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern,text,re.IGNORECASE)
        if match:
            skills.append(skill)
    return skills

def extract_education_from_resume(text):
    education= []
    education_words = [
        "High School", "Diploma", "Associate Degree", "Bachelor's Degree", "Master's Degree",
        "Doctorate", "PhD", "MBA", "BS", "BA", "MA", "MS", "MD", "Juris Doctor", "JD",
        "Engineering", "Computer Science", "Mathematics", "Physics", "Chemistry", "Biology",
        "Economics", "Business Administration", "Psychology", "Sociology", "Anthropology",
        "Political Science", "History", "Literature", "Philosophy", "Linguistics", "Education",
        "Fine Arts", "Nursing", "Health Sciences", "Public Health", "Environmental Science",
        "Law", "Medicine", "Dentistry", "Pharmacy", "Veterinary Medicine", "Agriculture",
        "Architecture", "Urban Planning", "Library Science", "Social Work", "Communications",
        "Journalism", "Media Studies", "Performing Arts", "Visual Arts", "Graphic Design",
        "Industrial Design", "Interior Design", "Fashion Design", "Culinary Arts", "Hospitality Management",
        "Tourism", "Real Estate", "Supply Chain Management", "Project Management", "Human Resources Management",
        "Information Technology", "Cybersecurity", "Data Science", "Statistics", "Operations Research",
        "Finance", "Accounting", "Marketing", "International Business", "Entrepreneurship",
        "Public Administration", "Nonprofit Management", "Religious Studies", "Theology", "Divinity",
        "Music", "Theater", "Dance", "Film", "Video Production", "Photography", "Creative Writing",
        "Technical Writing", "Comparative Literature", "Cognitive Science", "Neuroscience",
        "Bioinformatics", "Genetics", "Molecular Biology", "Biochemistry", "Microbiology",
        "Immunology", "Environmental Engineering", "Civil Engineering", "Mechanical Engineering",
        "Electrical Engineering", "Chemical Engineering", "Aerospace Engineering", "Materials Science",
        "Biomedical Engineering", "Industrial Engineering", "Systems Engineering", "Software Engineering",
        "Computer Engineering", "Robotics", "Artificial Intelligence", "Machine Learning",
        "Natural Language Processing", "Renewable Energy", "Sustainable Development", "Urban Studies",
        "Public Policy", "International Relations", "Peace and Conflict Studies", "Global Studies",
        "Asian Studies", "European Studies", "Latin American Studies", "African Studies",
        "Middle Eastern Studies", "Indigenous Studies", "Women’s Studies", "Gender Studies",
        "Sexuality Studies", "Disability Studies", "Cultural Studies", "American Studies"
    ]
    for word in education_words:
        pattern = r"(?i)\b{}\b".format(re.escape(word))
        match = re.search(pattern,text)
        if match:
            education.append(match.group())
    return education


#routes path
@app.route('/')
def resume():
    return render_template('Resume.html')

@app.route('/pred',methods=["POST"])
def pred():
        if 'resume' in request.files:
            file = request.files['resume']
            filename = file.filename

            if filename.endswith('.pdf'):
                text = pdf_to_text(file)
            elif filename.endswith('.txt'):
                text = file.read().decode('utf-8')
            else:
                return render_template('Resume.html',message="INVALID file format")

            extracted_name = extract_name_from_resume(text)
            extracted_phone = extract_contact_number_from_resume(text)
            extracted_email = extract_email_from_resume(text)
            extracted_skills = extract_skills_from_resume(text)
            extracted_education = extract_education_from_resume(text)

            predicted_cat = predict_category(text)
            recommended_job = job_recommendation(text)
            return render_template('resume.html',predicted_cat=predicted_cat,recommended_job=recommended_job,name=extracted_name,phone=extracted_phone,email=extracted_email,skills=extracted_skills,education=extracted_education)

#python main
if __name__=='__main__':
    app.run(debug=True)