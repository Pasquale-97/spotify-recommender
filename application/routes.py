from application import app
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template, redirect, url_for
import pandas as pd
import sys

from model import recommend_songs

spotify_df = pd.read_csv("/Users/pasqualeiuliano/Documents/AI/spotify/data/data.csv")

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return(render_template('index.html'))

    if request.method == 'POST':
        song = request.form['song']
        
        input_variables = pd.DataFrame([song])
        
        # x = 'Cupid\'s Chokehold / Breakfast in America - Radio Mix'
        prediction = recommend_songs([{'name': input_variables, 'year': 2007}],  spotify_df)

        return render_template('index.html',
                            original_input={'Song':song},
                            result=prediction)

if __name__ == '__main__':
    app.run()
