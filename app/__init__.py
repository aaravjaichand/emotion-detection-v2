from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Configure app
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
    
    # Register blueprints
    from app.routes import main
    app.register_blueprint(main)
    
    return app 