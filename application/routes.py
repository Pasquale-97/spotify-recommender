from application import app
from flask import Flask
from flask import request
from flask import jsonify
from flask import render_template, redirect, url_for
import pandas as pd
import sys

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
            list_of_songs.append(dictionary_song)

            return render_template('index.html', result=list_of_songs)

        if request.form['submit_button'] == 'Create Playlist':
            song = request.form['song']

            if not list_of_songs:
                message = "Please enter at least one song."
                return render_template('index.html', result=message)
        
            # input_variables = pd.DataFrame([song])
            # x = 'Cupid\'s Chokehold / Breakfast in America - Radio Mix'
            # [{'name': song}]

            prediction = recommend_songs(list_of_songs,  spotify_df)
        
            return render_template('index.html',
                            result=prediction)
        
        if request.form['submit_button'] == 'Clear':
            list_of_songs.clear()

            return render_template('index.html', result=list_of_songs)

if __name__ == '__main__':
    app.run()
