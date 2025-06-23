{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "9fc78ba5-0028-402f-9c38-5b8d2b9c013f",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pickle\n",
    "import joblib\n",
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "import time\n",
    "import pandas\n",
    "import os\n",
    "from flask import Flask, request, jsonify, render_template\n",
    "\n",
    "app = Flask(__name__)\n",
    "model = pickle.load(open('G:/AIBM/ML projects/Traffic_volume/model.pkl', 'rb'))\n",
    "scale = pickle.load(open('C:/Users/SmartbridgePC/Desktop/AIML/Guided projects/scale.pkl','rb'))\n",
    "\n",
    "@app.route('/')  # route to display the home page\n",
    "def home():\n",
    "    return render_template('index.html')  # rendering the home page\n",
    "\n",
    "@app.route('/predict', methods=[\"POST\",\"GET\"])  # route to show the predictions in a web UI\n",
    "def predict():\n",
    "    # reading the inputs given by the user\n",
    "    input_feature = [float(x) for x in request.form.values()]\n",
    "    features_values = np.array([input_feature])\n",
    "    names = [['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day',\n",
    "              'hours', 'minutes', 'seconds']]\n",
    "    \n",
    "    data = pandas.DataFrame(features_values, columns=names)\n",
    "    data = scale.fit_transform(data)\n",
    "    data = pandas.DataFrame(data, columns=names)\n",
    "\n",
    "    # predictions using the loaded model file\n",
    "    prediction = model.predict(data)\n",
    "    print(prediction)\n",
    "    text = \"Estimated Traffic Volume is :\"\n",
    "    return render_template(\"index.html\", prediction_text = text + str(prediction))\n",
    "\n",
    "# showing the prediction results in a UI\n",
    "if __name__ == \"__main__\":\n",
    "    # app.run(host='0.0.0.0', port=8000, debug=True)  # running the app\n",
    "    port = int(os.environ.get('PORT', 5000))\n",
    "    app.run(port=port, debug=True, use_reloader=False)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
