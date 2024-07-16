
# EMOTION DETECTION TO UPGRADE CUSTOMER EXPERIENCE

Growing technological development business platforms have shifted to web and cloudbased environments. Every customer needs privacy with respect to the purchases and services they seek. All business centers provide customer service support to clarify
their queries in services or products. The present feedback system uses QA which ends in a biased feedback that leads to inaccurate customer experience. Interaction between the customer and executive gets affected by various reasons including inappropriate executive, extended wait time, etc. This project provides a framework that identifies the mood/emotion of the customer based on the initial chat query in a chat application and
then uses sentiment analysis and routes the request to the appropriate technical expertto solve the issue which in turn improves the customer experience. After the call is
established, using speech recognition the system identifies the emotion of the customerfor the clarification/service provided by the person and grades them automatically. This framework for improving customer experience through sentiment analysis and emotion detection involves two phases: emotion-based call routing and auto-grading of service professionals.

The project proposes a Convolutional Neural Network(CNN) based architecture for Speech Emotion Recognition and classifies speech into angry, neutral, disappointed or happy. The outcome of this proposed system is to upgrade the customer experience by
analyzing the calls at the customer care center. This system can also be used in various fields other than feedback systems like in the diagnosis of physiological disorders and counseling as emotion is an important topic in psychology and neuroscience.
## Contributors

- [@Sowmiyanarayan Selvam](https://github.com/SowmiSelvam)
- [@Abisheck Kathirvel](https://github.com/abisheckkathir)
- [@Apoorvaa S Raghavan](https://github.com/Apoorvaa27)
- [@Arvind Thangaraj](https://github.com/thangarajarvind)
## Built With

[![Python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)

[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/en/3.0.x/)

[![MySQL](https://img.shields.io/badge/MySQL-005C84?style=for-the-badge&logo=mysql&logoColor=white)](https://www.mysql.com/)

![Tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
## Documentation

For comprehensive information about the project, please refer to the [Documentation](https://github.com/thangarajarvind/Emotion-detection-to-upgrade-customer-experience/blob/main/Emotion_detection_to_upgrade_Customer_Experience%20(3).pdf)


## Demo

![](https://github.com/thangarajarvind/Emotion-detection-to-upgrade-customer-experience/blob/main/Emotion_gif.gif)

For a detailed demonstration of the project, please watch the [Demo Video](https://drive.google.com/file/d/144hzQKfIDB_q-VB_LpS7AcGKMOVKGapG/view?usp=sharing)
## Support

For support, email thangarajarvind@gmail.com or DM via [@Arvind Thangaraj](https://www.linkedin.com/in/arvind-thangaraj/)


## Lessons Learned

To sum up the overall workflow of our project, the system starts with the user entering a query they are with in a form and submitting it. As the session starts the submitted query is pre-processed and sent to Long Short-Term Memory(LSTM) classifier. In the pre-processing stage, the text is cleaned removing punctuation and numbers. The pre-processed text is sent as input to the LSTM classifier to identify the emotion. The
identified emotion is used as input to the call routing module which will identify the appropriate technical expert to solve the issue which in turn improves the customer experience. After the call is established, the speech signals are pre-processed and the required features are extracted and then sent to MLP classifier which does speech recognition and thus the system identifies the emotion of the customer for the clarification/service provided by the person and grades them automatically. In feature extraction, Mfcc, Mel, and Chroma features are extracted from the sound file and stored in hstack. From the data set, the required emotions are selected and the MLP model is trained using the features extracted. Hyper-parameters were tweaked to get higher accuracy. This classifier returns the emotion of the customer during the call at certain intervals which will be used in the Integration module to automatically grade the customer service personnel.

