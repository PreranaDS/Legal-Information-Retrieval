# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 08:27:17 2022

@author: Toshiba
"""

import sys
import json
import os
import glob
import re
import  numpy as np
import pandas as pd
from flask import Flask,request,redirect,url_for,Response
from flask import render_template
import speech_recognition as sr
from legalinforetrieval import retrieve 
from speech import voiceconversion

xy=""

Image_FOLDER = os.path.join('static', 'images')
recieved_audio = sr.Recognizer()
df=pd.read_csv('./SIH-Processed.csv')
app=Flask(__name__)
app.config['UPLOAD_FOLDER'] = Image_FOLDER

@app.route('/', methods=['GET','POST'])
def index():
   filename = os.path.join(app.config['UPLOAD_FOLDER'], 'log.jpg')
   return render_template('index.html',user_image = filename)

@app.route('/_autocomplete', methods=['GET'])
def autocomplete():
    auto = list(df['case name'])
    print(auto)    
    return Response(json.dumps(auto), mimetype='application/json')




@app.route('/search',methods=['GET','POST'])
def search():
    if request.form.get('speak') == 'speak out':
        global xy
        xy=voiceconversion()
        
        
        c=retrieve(xy)
        if type(c)==str:
            pass
        else:
            c=c[['case name','document_url','score']]
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'log.jpg')
        return render_template('index.html',res=c,user_image = filename)
        
    elif xy!=None:
           
            x=request.form.get('query')
            court=request.form.get('court_name')
            judge=request.form.get('judge')
            date=request.form.get('date')
            res= retrieve(x)
            if type(res)==str:
                cases=res
            else:
                if court!='Select a court':
                    res=res[res['court']==court]
                if judge:
                    res=res[res['bench']==judge]
                if date!='Select Year':
                    res=res[res['case year']==int(date)]
        
                cases=res.head(5)[['case name','document_url','score']]
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'log.jpg')
            #pd.set_option('display.max_columns', None)
            #pd.set_option('display.max_colwidth', None)
            return render_template('index.html', res=cases,user_image = filename)


    
if __name__ == '__main__':
    app.run(port=3001)