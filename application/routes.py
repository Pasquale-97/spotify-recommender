from application import app
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template, redirect, url_for
import pandas as pd
import sys
from flask_cors import CORS, cross_origin
import re

from model import recommend_songs

spotify_df = pd.read_csv("/Users/pasqualeiuliano/Documents/AI/spotify/data/data.csv")
list_of_songs = []

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return(render_template('index.html'))
        

    if request.method == 'POST':
        if request.form['submit_button'] == 'Add Song':
            song = request.form['song']
            dictionary_song = {'name': song}
            if dictionary_song not in list_of_songs:
                list_of_songs.append(dictionary_song)
            
            filter_songs = [re.sub(r"[\[\]\'\"]", "", d['name']) for d in list_of_songs]
            
            return render_template('index.html', input_songs=filter_songs)
            

        if request.form['submit_button'] == 'Create Playlist':
            song = request.form['song']

            if not list_of_songs:
                message = "Please enter at least one song."
                return render_template('index.html', result=message)

            prediction = recommend_songs(list_of_songs, spotify_df)
            order_prediction = sorted(prediction, key=lambda k: k['popularity'], reverse=True) 

            headings = ("Songs", "Artists", "Year", "Popularity")
            
            tuple_of_songs = [(d['name'], re.sub(r"[\[\]\'\"]", "", d['artists']),d['year'],d['popularity']) for d in order_prediction]

            
            return render_template('index.html',
                            headings = headings,
                            result=tuple_of_songs)
        
        if request.form['submit_button'] == 'Clear':
            list_of_songs.clear()

            return render_template('index.html', result=list_of_songs)

if __name__ == '__main__':
    app.run()
