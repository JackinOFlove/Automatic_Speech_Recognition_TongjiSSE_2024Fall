# import create_app function here
from app import create_app
from flask_cors import CORS

app = create_app()

# allow all cross-domain requests in development environment
CORS(app, supports_credentials=True)

if __name__ == "__main__":
    # allow all domains to access
    app.run(
        debug=True,  
        host='0.0.0.0',  
        port=5000,  
        threaded=True  
    )