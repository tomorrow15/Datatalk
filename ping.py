from flask import Flask

app  = Flask('ping')
@app.route('/path',methods=['GET'])
def ping():
    return "REVISION"

app.run(debug=True,host='0.0.0.0',port=9696)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=9696)