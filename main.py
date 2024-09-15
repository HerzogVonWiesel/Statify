# create streamlit app with plotly graphs
import streamlit as st
import altair as alt
import pandas as pd
import numpy
import requests
import os
import glob
import yaml
import json
import geocoder
import time
from collections import deque
from vega_datasets import data

DEBUG = False

st.set_page_config(layout="wide")

# load API keys
with open("SPOT_API.yaml", "r") as f:
    config = yaml.safe_load(f)
CLIENT_ID = config["CLIENT_ID"]
CLIENT_SECRET = config["CLIENT_SECRET"]

# Cached Data
CACHE_ARTISTS = "CACHE/artists.parquet"
if os.path.exists(CACHE_ARTISTS):
    artist_cache_df = pd.read_parquet(CACHE_ARTISTS)
else:
    artist_cache_df = pd.DataFrame(columns=['artist_id', 'name', 'genres', 'followers', 'popularity'])
CACHE_TRACKS = "CACHE/tracks.parquet"
if os.path.exists(CACHE_TRACKS):
    tracks_cache_df = pd.read_parquet(CACHE_TRACKS)
else:
    tracks_cache_df = pd.DataFrame(columns=['track_id', 'name', 'artists_ids', 'album_id', 'explicit', 'duration_ms', 'popularity'])
CACHE_ALBUMS = "CACHE/albums.parquet"
if os.path.exists(CACHE_ALBUMS):
    albums_cache_df = pd.read_parquet(CACHE_ALBUMS)
else:
    albums_cache_df = pd.DataFrame(columns=['album_id', 'name', 'artists_ids', 'tracks_ids', 'release_date', 'total_tracks', 'genres', 'popularity'])

# generate access token
AUTH_URL = 'https://accounts.spotify.com/api/token'
auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
})
auth_response_data = auth_response.json()
access_token = auth_response_data['access_token']
headers = {'Authorization': 'Bearer {token}'.format(token=access_token)}
BASE_URL = 'https://api.spotify.com/v1/'

multi_word_genres = ['hip hop', 'boy band', 'movie tunes', 'new jersey', 'east coast', 'a cappella']
def split_genres(genre_list):
    split_list = []
    #print(genre_list)
    # if genre_list is nan, return empty list
    if isinstance(genre_list, float):
        return []
    for genre in genre_list:
        # Extract multi-word genres first
        for multi_word_genre in multi_word_genres:
            if multi_word_genre in genre:
                split_list.append(multi_word_genre)
                # Remove the found multi-word genre from the genre string
                genre = genre.replace(multi_word_genre, '')
        # Split the remaining genre string by spaces and add them
        split_list.extend(genre.split())
    # Remove any empty strings in case there were extra spaces
    return [genre for genre in split_list if genre]

def get_track_id(uri):
    return uri.split(":")[-1]  # Spotify URIs follow "spotify:track:{track_id}"

def get_data_from_spotify(id_type, id):
    if id_type == "artist":
        url = f"{BASE_URL}artists/{id}"
    elif id_type == "track":
        url = f"{BASE_URL}tracks/{id}"
    elif id_type == "album":
        url = f"{BASE_URL}albums/{id}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 429:
        time_to_wait = int(response.headers['Retry-After'])
        print(f"Rate limit exceeded. Waiting for {time_to_wait} seconds...")
        time.sleep(time_to_wait)
        return get_data_from_spotify(id_type, id)
    else:
        print("Error:", response.status_code)
        return None
    
def get_spotify_batch_data(id_type, id_list):
    if id_type == "track":
        url = f"{BASE_URL}tracks?ids=" + "%2C".join(id_list)
    elif id_type == "artist":
        url = f"{BASE_URL}artists?ids=" + "%2C".join(id_list)
    elif id_type == "album":
        url = f"{BASE_URL}albums?ids=" + "%2C".join(id_list)
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 429:
        time_to_wait = int(response.headers['Retry-After'])
        print(f"Rate limit exceeded. Waiting for {time_to_wait} seconds...")
        with st.spinner(f"Rate limit exceeded. Waiting for {time_to_wait} seconds..."):
            time.sleep(time_to_wait)
        return get_spotify_batch_data(id_type, id_list)
    else:
        st.error("Error while getting url: " + url + " with status code: " + str(response.status_code))
        return None

def get_artist_data(artist_id):
    artist_data = get_data_from_spotify("artist", artist_id)
    artist_data = {key: artist_data[key] for key in ['name', 'genres', 'followers', 'popularity']}
    if isinstance(artist_data['genres'], float):
        artist_data['genres'] = []
    artist_data['followers'] = artist_data['followers']['total']
    artist_data['artist_id'] = artist_id
    
    artist_cache_df.loc[len(artist_cache_df)] = artist_data
    artist_cache_df.to_parquet(CACHE_ARTISTS)
    return artist_data
    
def get_track_data(track_id):
    track_data = get_data_from_spotify("track", track_id)
    track_data = {key: track_data[key] for key in ['name', 'artists', 'explicit', 'album', 'duration_ms', 'popularity']}
    track_data['artists_ids'] = [artist['id'] for artist in track_data['artists']]
    track_data['album_id'] = track_data['album']['id']
    track_data['track_id'] = track_id
    track_data.pop('album')
    track_data.pop('artists')
    
    tracks_cache_df.loc[len(tracks_cache_df)] = track_data
    tracks_cache_df.to_parquet(CACHE_TRACKS)
    return track_data
    
def get_album_data(album_id):
    album_data = get_data_from_spotify("album", album_id)
    album_data = {key: album_data[key] for key in ['name', 'artists', 'tracks', 'release_date', 'total_tracks', 'genres', 'popularity']}
    album_data['artists_ids'] = [artist['id'] for artist in album_data['artists']]
    album_data['tracks_ids'] = [track['id'] for track in album_data['tracks']['items']]
    album_data['album_id'] = album_id
    album_data.pop('tracks')
    album_data.pop('artists')
    
    albums_cache_df.loc[len(albums_cache_df)] = album_data
    albums_cache_df.to_parquet(CACHE_ALBUMS)
    return album_data

def get_col_from_artist(artist_id, col):
    if artist_id in artist_cache_df['artist_id'].values:
        cached_data = artist_cache_df.loc[artist_cache_df['artist_id'] == artist_id, col].values[0]
        return cached_data
    return get_artist_data(artist_id)[col]

def get_col_from_track(track_id, col):
    if track_id in tracks_cache_df['track_id'].values:
        cached_data = tracks_cache_df.loc[tracks_cache_df['track_id'] == track_id, col].values[0]
        return cached_data
    return get_track_data(track_id)[col]

def get_col_from_album(album_id, col):
    if album_id in albums_cache_df['album_id'].values:
        cached_data = albums_cache_df.loc[albums_cache_df['album_id'] == album_id, col].values[0]
        return cached_data
    return get_album_data(album_id)[col]

def fill_related_data_from_track(track_id):
    artists_ids = get_col_from_track(track_id, 'artists_ids')
    album_id = get_col_from_track(track_id, 'album_id')
    for artist_id in artists_ids:
        get_col_from_artist(artist_id, 'name')
    get_col_from_album(album_id, 'name')

def fill_cache(df):
    global tracks_cache_df
    global artist_cache_df
    global albums_cache_df
    progress_text = "Gathering data from Spotify. Please wait."
    my_bar = st.progress(0, text=progress_text)
    # remove lines in df that have no uri
    df = df.dropna(subset=['uri'])
    track_ids = df['uri'].apply(get_track_id).unique()
    existing_track_ids = set(tracks_cache_df['track_id'])
    new_track_ids = [track_id for track_id in track_ids if track_id not in existing_track_ids]
    # get datas in batches of 50
    batch_size = 50
    for i in range(0, len(new_track_ids), batch_size):
        batch_track_ids = new_track_ids[i:i+batch_size]
        batch_track_data = get_spotify_batch_data("track", batch_track_ids)
        batch_track_df = pd.DataFrame(batch_track_data['tracks'])
        batch_track_df['artists_ids'] = batch_track_df['artists'].apply(lambda x: [artist['id'] for artist in x])
        batch_track_df['album_id'] = batch_track_df['album'].apply(lambda x: x['id'])
        batch_track_df['track_id'] = batch_track_df['id']
        # drop all columns except the ones we want to cache
        batch_track_df = batch_track_df[['track_id', 'name', 'artists_ids', 'album_id', 'explicit', 'duration_ms', 'popularity']]
        tracks_cache_df = pd.concat([tracks_cache_df, batch_track_df])
        tracks_cache_df.to_parquet(CACHE_TRACKS)
        my_bar.progress(min(((i + batch_size) / len(new_track_ids))/3, 1.0))
    
    existing_artist_ids = set(artist_cache_df['artist_id'])
    new_artist_ids = set(tracks_cache_df['artists_ids'].explode().unique()) - existing_artist_ids
    new_artist_ids = list(new_artist_ids)
    for i in range(0, len(new_artist_ids), batch_size):
        batch_artist_ids = new_artist_ids[i:i+batch_size]
        batch_artist_data = get_spotify_batch_data("artist", batch_artist_ids)
        batch_artist_df = pd.DataFrame(batch_artist_data['artists'])
        batch_artist_df['artist_id'] = batch_artist_df['id']
        batch_artist_df['followers'] = batch_artist_df['followers'].apply(lambda x: x['total'])
        # drop all columns except the ones we want to cache
        batch_artist_df = batch_artist_df[['artist_id', 'name', 'genres', 'followers', 'popularity']]
        artist_cache_df = pd.concat([artist_cache_df, batch_artist_df])
        artist_cache_df.to_parquet(CACHE_ARTISTS)
        my_bar.progress(min(((i + batch_size) / len(new_artist_ids)) * 1/3 + 1/3, 1.0))

    existing_album_ids = set(albums_cache_df['album_id'])
    new_album_ids = set(tracks_cache_df['album_id'].unique()) - existing_album_ids
    new_album_ids = list(new_album_ids)
    for i in range(0, len(new_album_ids), 20):
        batch_album_ids = new_album_ids[i:i+20]
        batch_album_data = get_spotify_batch_data("album", batch_album_ids)
        batch_album_df = pd.DataFrame(batch_album_data['albums'])
        batch_album_df['artists_ids'] = batch_album_df['artists'].apply(lambda x: [artist['id'] for artist in x])
        batch_album_df['tracks_ids'] = batch_album_df['tracks'].apply(lambda x: [track['id'] for track in x['items']])
        batch_album_df['album_id'] = batch_album_df['id']
        batch_album_df['release_date'] = batch_album_df['release_date'].apply(lambda x: x[:10])
        # drop all columns except the ones we want to cache
        batch_album_df = batch_album_df[['album_id', 'name', 'artists_ids', 'tracks_ids', 'release_date', 'total_tracks', 'genres', 'popularity']]
        albums_cache_df = pd.concat([albums_cache_df, batch_album_df])
        albums_cache_df.to_parquet(CACHE_ALBUMS)
        my_bar.progress(min(1.0, ((i + 20) / len(new_album_ids))*1/3 + 2/3))
    my_bar.empty()

def add_genres_to_tracks(df_stream, df_artist_cache, simplify_genres=False):
    df_artist_cache_grouped = df_artist_cache.groupby('name', as_index=False).agg({
        'genres': lambda x: list(set([genre for sublist in x for genre in sublist]))
    })
    df_merged = df_stream.merge(df_artist_cache_grouped[['name', 'genres']], left_on='artistName', right_on='name', how='left')
    df_merged = df_merged.drop(columns=['name'])
    # replace NaNs in 'genres' column with empty list
    #st.write(df_merged.head())
    df_merged['genres'] = df_merged['genres'].apply(lambda x: x if not isinstance(x, float) else [])
    if simplify_genres:
        df_merged['genres'] = df_merged['genres'].apply(split_genres)
    return df_merged

def top_artists_playtime_and_plays(df_stream, filter_short_streams=True):
    if filter_short_streams:
        df_stream = df_stream[df_stream['msPlayed'] >= 10000]
    artist_stats = df_stream.groupby('artistName').agg(
        playtime_minutes=('msPlayed', lambda x: x.sum() / 1000 / 60),  # Sum playtime in minutes
        plays=('artistName', 'size')  # Count number of plays
    ).reset_index()
    return artist_stats

def top_artists_playtime(artist_stats, number_of_artists):
    top_artists = artist_stats.sort_values('playtime_minutes', ascending=False).head(number_of_artists)
    return top_artists

def top_artists_plays(artist_stats, number_of_artists):
    top_artists = artist_stats.sort_values('plays', ascending=False).head(number_of_artists)
    return top_artists

def top_artists_xy(artist_stats, number_of_artists):
    top_artists_playtime_df = top_artists_playtime(artist_stats, number_of_artists)
    top_artists_plays_df = top_artists_plays(artist_stats, number_of_artists)
    top_artists = pd.merge(top_artists_playtime_df, top_artists_plays_df, on='artistName', how='outer')
    # Replace NaNs (if there are any) in 'plays' and 'playtime_minutes' columns from both sides
    top_artists['playtime_minutes'] = top_artists['playtime_minutes_x'].combine_first(top_artists['playtime_minutes_y'])
    top_artists['plays'] = top_artists['plays_x'].combine_first(top_artists['plays_y'])
    top_artists = top_artists.drop(columns=['playtime_minutes_x', 'playtime_minutes_y', 'plays_x', 'plays_y'])
    return top_artists

def top_tracks_playtime_and_plays(df_stream, filter_short_streams=True):
    if filter_short_streams:
        df_stream = df_stream[df_stream['msPlayed'] >= 10000]
    track_stats = df_stream.groupby(['trackName', 'artistName']).agg(
        playtime_minutes=('msPlayed', lambda x: x.sum() / 1000 / 60),  # Sum playtime in minutes
        plays=('trackName', 'size')  # Count number of plays
    ).reset_index()
    return track_stats

def top_tracks_playtime(track_stats, number_of_tracks, ascending=False):
    top_tracks = track_stats.sort_values('playtime_minutes', ascending=ascending).head(number_of_tracks)
    return top_tracks

def top_tracks_plays(track_stats, number_of_tracks, ascending=False):
    top_tracks = track_stats.sort_values('plays', ascending=ascending).head(number_of_tracks)
    return top_tracks

def top_tracks_xy(track_stats, number_of_tracks, ascending=False):
    top_tracks_playtime_df = top_tracks_playtime(track_stats, number_of_tracks, ascending)
    top_tracks_plays_df = top_tracks_plays(track_stats, number_of_tracks, ascending)
    top_tracks = pd.merge(top_tracks_playtime_df, top_tracks_plays_df, on=['trackName', 'artistName'], how='outer')
    # Replace NaNs (if there are any) in 'plays' and 'playtime_minutes' columns from both sides
    top_tracks['playtime_minutes'] = top_tracks['playtime_minutes_x'].combine_first(top_tracks['playtime_minutes_y'])
    top_tracks['plays'] = top_tracks['plays_x'].combine_first(top_tracks['plays_y'])
    top_tracks = top_tracks.drop(columns=['playtime_minutes_x', 'playtime_minutes_y', 'plays_x', 'plays_y'])
    return top_tracks

def track_genres(df_with_track_uri, simplify_genres, is_extended_history=False, group_by_quarter=False):
    if simplify_genres:
        df_with_track_uri['genres'] = df_with_track_uri['genres'].apply(split_genres)
    df_genres_exploded = df_with_track_uri.explode('genres')
    if is_extended_history:
        subset = ['uri', 'genres', 'endTime']
        if group_by_quarter:
            df_genres_exploded['endTime'] = df_genres_exploded['endTime'].dt.to_period('Q')
    else:
        subset = ['track_id', 'genres']
    df_genres_exploded = df_genres_exploded.drop_duplicates(subset=subset)
    if group_by_quarter:
        df_genre_counts = df_genres_exploded.groupby(['genres', 'endTime']).size().reset_index(name='genre_count')
    else:
        df_genre_counts = df_genres_exploded.groupby(['genres']).size().reset_index(name='genre_count')
    #print(df_genre_counts.head())
    df_genre_counts = df_genre_counts.sort_values('genre_count', ascending=False).reset_index(drop=True)

    return df_genre_counts


def tracks_to_save(track_stats, df_lib_tracks, number_of_tracks):
    # Find tracks that have been played often / many times but are not in the library
    df_stream_unsaved = track_stats[~track_stats['trackName'].isin(df_lib_tracks['track'])]
    top_unsaved_tracks = top_tracks_plays(df_stream_unsaved, number_of_tracks)
    return top_unsaved_tracks

def tracks_to_delete(df_stream, df_lib_tracks, number_of_tracks):
    # Find tracks that have been skipped often / many times but are in the library
    track_stats = top_tracks_playtime_and_plays(df_stream, filter_short_streams=False)
    df_stream_saved = track_stats[track_stats['trackName'].isin(df_lib_tracks['track'])]
    df_stream_saved['avg_playtime'] = df_stream_saved['playtime_minutes'] / df_stream_saved['plays']
    df_stream_saved = df_stream_saved[df_stream_saved['avg_playtime'] < 0.5]
    top_saved_tracks = df_stream_saved.sort_values('plays', ascending=False).head(number_of_tracks)
    return top_saved_tracks

#def top_hated_songs(df_stream, number_of_tracks):
    
#def streamed_closest_neighbors

def stream_locations(df_stream):
    top_countries = df_stream.groupby('conn_country').size().reset_index(name='stream_count').sort_values(by='stream_count', ascending=False)

    # Load the country codes CSV
    country_codes = pd.read_csv("CACHE/countries_codes.csv", quotechar='"', skipinitialspace=True)
    
    # Ensure numeric_code is numeric
    country_codes['numeric_code'] = pd.to_numeric(country_codes['Numeric code'], errors='coerce')

    # Merge the top_countries with the country codes to add the 'numeric_code' column
    top_countries = pd.merge(top_countries, country_codes[['Alpha-2 code', 'numeric_code']], 
                             left_on='conn_country', right_on='Alpha-2 code', how='outer')
    # fill stream count nas with 0
    top_countries['stream_count'] = top_countries['stream_count'].fillna(0)

    # Drop the 'Alpha-2 code' column as it's redundant now
    top_countries = top_countries.drop(columns=['Alpha-2 code'])
    return top_countries

####################################################################################################
#########                                 Streamlit UI                                 #############
####################################################################################################
st.title("Statify: Spotify Listening History Analysis")

st.sidebar.title("Settings")
stats, meta, rec = st.tabs(["Stats", "Meta", "Recommendations"])
stat_a = stats.container()
meta_a = meta.container()
meta1, meta2 = meta.columns(2)

is_extended_history = st.sidebar.checkbox("Use extended history", value=False)

# load spotify listening history
glob_string = "data/StreamingHistory*.json" if not is_extended_history else "data/Streaming_History*.json"
df_stream = pd.DataFrame()
for file in glob.glob(glob_string):
    if DEBUG: # if debug, only take first 20 streams
        df_stream = pd.concat([df_stream, pd.read_json(file).head(20)])
    else:    
        df_stream = pd.concat([df_stream, pd.read_json(file)])
        print("DF_LENGHT: " + str(len(df_stream)))
# extended history has column names "ts", "master_metadata_track_name", "master_metadata_album_artist_name", "master_metadata_album_album_name" which we will rename to the other standard
if is_extended_history:
    df_stream = df_stream.rename(columns={"ts": "endTime", "ms_played": "msPlayed", "spotify_track_uri":"uri", "master_metadata_track_name": "trackName", "master_metadata_album_artist_name": "artistName", "master_metadata_album_album_name": "albumName"})
    # ts is formatted like "2020-08-02T15:27:43Z" but endTime is formatted like "2023-09-07 17:41"
    df_stream['endTime'] = pd.to_datetime(df_stream['endTime']).dt.strftime('%Y-%m-%d %H:%M:%S')
try:
    with open("data/YourLibrary.json") as f:
        library_data = json.load(f)
    df_lib_tracks = pd.DataFrame(library_data['tracks'])
    df_lib_albums = pd.DataFrame(library_data['albums'])
    df_lib_artists = pd.DataFrame(library_data['artists'])
    df_lib_banned_tracks = pd.DataFrame(library_data['bannedTracks'])
except FileNotFoundError:
    st.error("YourLibrary.json not found. Please download and place in the data folder.")

st.sidebar.divider()
df_stream['endTime'] = pd.to_datetime(df_stream['endTime'])
start_date = st.sidebar.date_input("Select start date:", min(df_stream['endTime']).date())
end_date = st.sidebar.date_input("Select end date:", max(df_stream['endTime']).date())
df_stream = df_stream[(df_stream['endTime'] >= pd.to_datetime(start_date)) & (df_stream['endTime'] <= pd.to_datetime(end_date))]
st.sidebar.divider()
show_by = st.sidebar.radio("Show by:", ["Plays", "Playtime"])
top_n = st.sidebar.slider("Number of top artists/tracks to show:", 5, 100, 10)
st.sidebar.divider()
simplify_genres = st.sidebar.checkbox("Simplify genres", value=False)


df_stream = add_genres_to_tracks(df_stream, artist_cache_df, simplify_genres)

# create df_podcast_streams from df_stream (has spotify_episode_uri != None)
if is_extended_history:
    df_podcast_streams = df_stream.dropna(subset=['spotify_episode_uri'])
    df_stream = df_stream.dropna(subset=['uri'])

##########                               Stats Tab                                        ##########

artist_brush = alt.selection_point(name='artist', fields=['artistName'], resolve='global')
genre_brush = alt.selection_point(name='genre', fields=['genres'], resolve='global')
country_brush = alt.selection_point(name='country', fields=['conn_country'], resolve='global')

try:
    selected_genres = [genre['genres'] for genre in st.session_state.selected_genre_data['selection']['genre']]
except AttributeError:
    selected_genres = []
try:
    print(st.session_state.selected_country_data['selection'])
    selected_countries = [country['conn_country'] for country in st.session_state.selected_country_data['selection']['country']]
except AttributeError:
    selected_countries = []

if selected_countries:
    df_stream_country_filtered = df_stream[df_stream['conn_country'].apply(lambda x: any([country in x for country in selected_countries]))]
else:
    df_stream_country_filtered = df_stream
if selected_genres:
    df_stream_filtered = df_stream_country_filtered[df_stream_country_filtered['genres'].apply(lambda x: any([genre in x for genre in selected_genres]))]
else:
    df_stream_filtered = df_stream_country_filtered

artists_stats_df = top_artists_playtime_and_plays(df_stream_filtered, filter_short_streams=True)
if show_by == "Plays":
    top_artists_df = top_artists_plays(artists_stats_df, top_n)
    axis_y = "plays"
    label_y = "Number of Plays"
elif show_by == "Playtime":
    top_artists_df = top_artists_playtime(artists_stats_df, top_n)
    axis_y = "playtime_minutes"
    label_y = "Total Playtime (minutes)"


stat_artists = stats.container()
stat_artist_top = stat_artists.container()
stat_artists_1, stat_artists_2 = stat_artists.columns(2)

stat_artist_top.caption("You can select artists in the bar chart to filter. Select multiple by holding down the shift key.")
stat_artists_1.subheader(f"Top Artists by {show_by}")
top_artists_chart = alt.Chart(top_artists_df).mark_bar().encode(
    x=alt.X('artistName:N', title=None, sort='-y', axis=alt.Axis(labelOverlap=False, labelAngle=-45)),  # Sort x-axis by number of plays
    y=alt.Y(f'{axis_y}:Q', title=label_y),
    opacity=alt.condition(artist_brush, alt.value(1.0), alt.value(0.5)),
    color=alt.value("#FF4B4B"),
    #tooltip=['artistName', 'plays']  # Tooltip for hover
).add_params(
    artist_brush
)
artist_data = stat_artists_1.altair_chart(top_artists_chart, on_select='rerun', use_container_width=True)
selected_artists = [artist['artistName'] for artist in artist_data['selection']['artist']]

top_artists_xy_df = top_artists_xy(artists_stats_df, top_n)
stat_artists_2.subheader(f"Top Artists by Playtime and Plays")
top_artists_xy_chart = alt.Chart(top_artists_xy_df).mark_circle().encode(
    x=alt.X('plays:Q', title='Number of Plays'),
    y=alt.Y('playtime_minutes:Q', title='Total Playtime (minutes)'),
    size=alt.Size('plays:Q', legend=None),
    color=alt.Color('playtime_minutes:Q', legend=None, scale=alt.Scale(scheme='reds')),
    #opacity=alt.condition(altair_brush, alt.value(1.0), alt.value(0.5)),
    tooltip=['artistName', 'plays', 'playtime_minutes'],
)
artist_data = stat_artists_2.altair_chart(top_artists_xy_chart, use_container_width=True)
# NOT SUPPORTED YET artist_data = st.altair_chart(top_artists_chart | top_artists_xy_chart, on_select='rerun', use_container_width=True)


stat_tracks = stats.container()
stat_tracks_1, stat_tracks_2 = stat_tracks.columns(2)

if selected_artists:
    df_stream_selected_artist = df_stream_filtered[df_stream_filtered['artistName'].isin(selected_artists)]
else:
    df_stream_selected_artist = df_stream_filtered
track_stats_df = top_tracks_playtime_and_plays(df_stream_selected_artist, filter_short_streams=True)
if show_by == "Plays":
    top_tracks_df = top_tracks_plays(track_stats_df, top_n)
elif show_by == "Playtime":
    top_tracks_df = top_tracks_playtime(track_stats_df, top_n)

stat_tracks_1.subheader(f"Top Tracks by {show_by}")
top_tracks_plays_chart = alt.Chart(top_tracks_df).mark_bar().encode(
    y=alt.X(f'{axis_y}:Q', title=label_y),
    x=alt.Y('trackName:N', title=None, sort='-y', axis=alt.Axis(labelOverlap=False, labelAngle=-45)),  # Sort y-axis by number of plays
    color=alt.value("#FF4B4B"),
)
stat_tracks_1.altair_chart(top_tracks_plays_chart, use_container_width=True)

top_tracks_xy_df = top_tracks_xy(track_stats_df, top_n)
stat_tracks_2.subheader(f"Top Tracks by Playtime and Plays")
top_tracks_chart = alt.Chart(top_tracks_xy_df).mark_circle().encode(
    x=alt.X('plays:Q', title='Number of Plays'),
    y=alt.Y('playtime_minutes:Q', title='Total Playtime (minutes)'),
    size=alt.Size('plays:Q', legend=None),
    color=alt.Color('playtime_minutes:Q', legend=None, scale=alt.Scale(scheme='reds')),
    tooltip=['trackName', 'artistName', 'plays', 'playtime_minutes'],
)
stat_tracks_2.altair_chart(top_tracks_chart, use_container_width=True)

if is_extended_history:
    meta_countries_a = meta.container()
    meta_countries = meta.container()
    meta_countries_1, meta_countries_2 = meta_countries.columns(2)
    meta_countries_a.subheader("Countries")
    country_counts = stream_locations(df_stream_filtered)
    world_map = alt.topo_feature(data.world_110m.url, 'countries')

    # Create an Altair choropleth map
    choropleth = alt.Chart(world_map).mark_geoshape().encode(
        color=alt.Color('stream_count:Q', scale=alt.Scale(type='symlog', scheme='reds')),
        opacity=alt.condition(country_brush, alt.value(1.0), alt.value(0.5)),
        tooltip=[alt.Tooltip('conn_country:N', title="Country"), alt.Tooltip('stream_count:Q', title="Stream Count")]
    ).transform_lookup(
        lookup='id',
        from_=alt.LookupData(country_counts, 'numeric_code', ['conn_country', 'stream_count'])
    ).project(
        'naturalEarth1'
    ).add_params(
        country_brush
    )
    meta_countries_a.altair_chart(choropleth, on_select='rerun', key='selected_country_data', use_container_width=True)



stat_genres = stats.container()
stat_genres_1, stat_genres_2 = stat_genres.columns(2)

if True:
    fill_cache(df_stream)

stat_genres_1.subheader("Genres")
if True and is_extended_history:
    genre_counts = track_genres(df_stream_country_filtered, simplify_genres, is_extended_history).head(50)
else:
    genre_counts = track_genres(df_lib_tracks, simplify_genres).head(50)
genre_counts_chart = alt.Chart(genre_counts).mark_arc().encode(
    theta=alt.Theta('genre_count:Q', stack=True),
    order={"field": "genre_count", "type": "quantitative", "sort": "descending"},
    color=alt.Color('genres:N', sort=alt.EncodingSortField(field='genre_count', order='descending')),
    opacity=alt.condition(genre_brush, alt.value(1.0), alt.value(0.5)),
    tooltip=['genres', 'genre_count']
).add_params(
    genre_brush
)
stat_genres_1.altair_chart(genre_counts_chart, on_select='rerun', key='selected_genre_data', use_container_width=True)
#selected_genres = [genre['genres'] for genre in st.session_state.selected_genre_data['selection']['genre']]
if is_extended_history:
    top_10_genres = genre_counts['genres'].head(10).tolist()
    genre_counts_quarterly = track_genres(df_stream_country_filtered, simplify_genres, is_extended_history, group_by_quarter=True)
    genre_counts_quarterly = genre_counts_quarterly[genre_counts_quarterly['genres'].isin(top_10_genres)]
    # endTime to string to display in altair chart
    genre_counts_quarterly['endTime'] = genre_counts_quarterly['endTime'].astype(str)
    stat_genres_2.subheader("Genres by Quarter")
    genre_counts_quarterly_chart = alt.Chart(genre_counts_quarterly).mark_line(interpolate="monotone").encode(
        x=alt.X('endTime:N', title='Quarter'),
        y='genre_count:Q',
        # sort by top_10_genres and the order they are in the list
        color=alt.Color('genres:N', sort=top_10_genres, legend=None),
        # make tooltip area bigger
        tooltip=['genres', 'genre_count', 'endTime:N']
    ).interactive()
    stat_genres_2.altair_chart(genre_counts_quarterly_chart, use_container_width=True)
else:
    stat_genres_2.warning("Be careful selecting genres when using the simple (not extended) streaming history. We can only consicder artists of whom you have songs saved in your library then!")


rec_a = rec.container()

rec_a.subheader("Tracks to Save")
tracks_to_save_df = tracks_to_save(track_stats_df, df_lib_tracks, top_n)
rec_a.dataframe(tracks_to_save_df, hide_index=True, use_container_width=True)

rec_a.subheader("Tracks to Delete")
tracks_to_delete_df = tracks_to_delete(df_stream, df_lib_tracks, top_n)
rec_a.dataframe(tracks_to_delete_df, hide_index=True, use_container_width=True)