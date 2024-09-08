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

st.set_page_config(layout="wide")

# load spotify listening history
df_stream = pd.DataFrame()
for file in glob.glob("data/StreamingHistory*.json"):
    df_stream = pd.concat([df_stream, pd.read_json(file)])
try:
    with open("data/YourLibrary.json") as f:
        library_data = json.load(f)
    df_lib_tracks = pd.DataFrame(library_data['tracks'])
except FileNotFoundError:
    st.error("YourLibrary.json not found. Please download and place in the data folder.")

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

def get_track_id(uri):
    return uri.split(":")[-1]  # Spotify URIs follow "spotify:track:{track_id}"

def get_artist_data(artist_id):
    artist_url = f"{BASE_URL}artists/{artist_id}"
    response = requests.get(artist_url, headers=headers)
    if response.status_code == 200:
        artist_data = response.json()
        artist_data = {key: artist_data[key] for key in ['name', 'genres', 'followers', 'popularity']}
        artist_data['followers'] = artist_data['followers']['total']
        artist_data['artist_id'] = artist_id
        
        artist_cache_df.loc[len(artist_cache_df)] = artist_data
        artist_cache_df.to_parquet(CACHE_ARTISTS)
        return artist_data
    else:
        print("Error:", response.status_code)
        return None
    
def get_track_data(track_id):
    track_url = f"{BASE_URL}tracks/{track_id}"
    response = requests.get(track_url, headers=headers)
    if response.status_code == 200:
        track_data = response.json()
        track_data = {key: track_data[key] for key in ['name', 'artists', 'explicit', 'album', 'duration_ms', 'popularity']}
        track_data['artists_ids'] = [artist['id'] for artist in track_data['artists']]
        track_data['album_id'] = track_data['album']['id']
        track_data['track_id'] = track_id
        track_data.pop('album')
        track_data.pop('artists')
        
        tracks_cache_df.loc[len(tracks_cache_df)] = track_data
        tracks_cache_df.to_parquet(CACHE_TRACKS)
        return track_data
    else:
        print("Error:", response.status_code)
        return None
    
def get_album_data(album_id):
    album_url = f"{BASE_URL}albums/{album_id}"
    response = requests.get(album_url, headers=headers)
    if response.status_code == 200:
        album_data = response.json()
        album_data = {key: album_data[key] for key in ['name', 'artists', 'tracks', 'release_date', 'total_tracks', 'genres', 'popularity']}
        album_data['artists_ids'] = [artist['id'] for artist in album_data['artists']]
        album_data['tracks_ids'] = [track['id'] for track in album_data['tracks']['items']]
        album_data['album_id'] = album_id
        album_data.pop('tracks')
        album_data.pop('artists')
        
        albums_cache_df.loc[len(albums_cache_df)] = album_data
        albums_cache_df.to_parquet(CACHE_ALBUMS)
        return album_data
    else:
        print("Error:", response.status_code)
        return None

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

def top_artists_playtime_and_plays(df_stream, filter_short_streams=True):
    if filter_short_streams:
        df_stream = df_stream[df_stream['msPlayed'] >= 10000]
    #df_stream_filtered['playtime_minutes'] = df_stream_filtered['msPlayed'] / 1000 / 60  # Convert ms to minutes
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

def top_tracks_playtime(track_stats, number_of_tracks):
    top_tracks = track_stats.sort_values('playtime_minutes', ascending=False).head(number_of_tracks)
    return top_tracks

def top_tracks_plays(track_stats, number_of_tracks):
    top_tracks = track_stats.sort_values('plays', ascending=False).head(number_of_tracks)
    return top_tracks

def top_tracks_xy(track_stats, number_of_tracks):
    top_tracks_playtime_df = top_tracks_playtime(track_stats, number_of_tracks)
    top_tracks_plays_df = top_tracks_plays(track_stats, number_of_tracks)
    top_tracks = pd.merge(top_tracks_playtime_df, top_tracks_plays_df, on=['trackName', 'artistName'], how='outer')
    # Replace NaNs (if there are any) in 'plays' and 'playtime_minutes' columns from both sides
    top_tracks['playtime_minutes'] = top_tracks['playtime_minutes_x'].combine_first(top_tracks['playtime_minutes_y'])
    top_tracks['plays'] = top_tracks['plays_x'].combine_first(top_tracks['plays_y'])
    top_tracks = top_tracks.drop(columns=['playtime_minutes_x', 'playtime_minutes_y', 'plays_x', 'plays_y'])
    return top_tracks

def library_genres(df_library):
    track_ids = [get_track_id(track['uri']) for track in df_library if 'uri' in track]


def tracks_to_save(track_stats, df_library, number_of_tracks):
    # Find tracks that have been played often / many times but are not in the library
    df_stream_unsaved = track_stats[~track_stats['trackName'].isin(df_library['trackName'])]
    top_unsaved_tracks = top_tracks_xy(track_stats, number_of_tracks)
    return top_unsaved_tracks

#def top_hated_songs(df_stream, number_of_tracks):
    
#def streamed_closest_neighbors

#def stream_locations(df_stream):

####################################################################################################
#########                                 Streamlit UI                                 #############
####################################################################################################
st.title("Statify: Spotify Listening History Analysis")

st.sidebar.title("Settings")
stats, meta, rec = st.tabs(["Stats", "Meta", "Recommendations"])
stat_a = stats.container()
stat1, stat2 = stats.columns(2)
meta_a = meta.container()
meta1, meta2 = meta.columns(2)
rec_a = rec.container()
rec1, rec2 = rec.columns(2)
# set start and end date
df_stream['endTime'] = pd.to_datetime(df_stream['endTime'])
start_date = st.sidebar.date_input("Select start date:", min(df_stream['endTime']).date())
end_date = st.sidebar.date_input("Select end date:", max(df_stream['endTime']).date())
df_stream = df_stream[(df_stream['endTime'] >= pd.to_datetime(start_date)) & (df_stream['endTime'] <= pd.to_datetime(end_date))]
# set whether to show by plays or by playtime
show_by = st.sidebar.radio("Show by:", ["Plays", "Playtime"])
top_n = st.sidebar.slider("Number of top artists/tracks to show:", 5, 20, 10)


##########                               Stats Tab                                        ##########

altair_brush = alt.selection_point(name='artist', fields=['artistName'], resolve='global')

artists_stats_df = top_artists_playtime_and_plays(df_stream, filter_short_streams=True)
if show_by == "Plays":
    top_artists_df = top_artists_plays(artists_stats_df, top_n)
    axis_y = "plays"
    label_y = "Number of Plays"
elif show_by == "Playtime":
    top_artists_df = top_artists_playtime(artists_stats_df, top_n)
    axis_y = "playtime_minutes"
    label_y = "Total Playtime (minutes)"

stat_a.caption("You can select artists in the bar chart to filter. Select multiple by holding down the shift key.")
stat1.subheader(f"Top Artists by {show_by}")
top_artists_chart = alt.Chart(top_artists_df).mark_bar().encode(
    x=alt.X('artistName:N', title=None, sort='-y', axis=alt.Axis(labelOverlap=False, labelAngle=-45)),  # Sort x-axis by number of plays
    y=alt.Y(f'{axis_y}:Q', title=label_y),
    opacity=alt.condition(altair_brush, alt.value(1.0), alt.value(0.5)),
    color=alt.value("#FF4B4B"),
    #tooltip=['artistName', 'plays']  # Tooltip for hover
).add_params(
    altair_brush
)
artist_data = stat1.altair_chart(top_artists_chart, on_select='rerun', use_container_width=True)
selected_artists = [artist['artistName'] for artist in artist_data['selection']['artist']]

top_artists_xy_df = top_artists_xy(artists_stats_df, top_n)
stat2.subheader(f"Top Artists by Playtime and Plays")
top_artists_xy_chart = alt.Chart(top_artists_xy_df).mark_circle().encode(
    x=alt.X('plays:Q', title='Number of Plays'),
    y=alt.Y('playtime_minutes:Q', title='Total Playtime (minutes)'),
    size=alt.Size('plays:Q', legend=None),
    color=alt.Color('playtime_minutes:Q', legend=None, scale=alt.Scale(scheme='reds')),
    #opacity=alt.condition(altair_brush, alt.value(1.0), alt.value(0.5)),
    tooltip=['artistName', 'plays', 'playtime_minutes'],
).add_params(
    altair_brush
)
artist_data = stat2.altair_chart(top_artists_xy_chart, use_container_width=True)
# NOT SUPPORTED YET artist_data = st.altair_chart(top_artists_chart | top_artists_xy_chart, on_select='rerun', use_container_width=True)

if selected_artists:
    df_stream_selected_artist = df_stream[df_stream['artistName'].isin(selected_artists)]
else:
    df_stream_selected_artist = df_stream
track_stats_df = top_tracks_playtime_and_plays(df_stream_selected_artist, filter_short_streams=True)
if show_by == "Plays":
    top_tracks_df = top_tracks_plays(track_stats_df, top_n)
elif show_by == "Playtime":
    top_tracks_df = top_tracks_playtime(track_stats_df, top_n)

stat1.subheader(f"Top Tracks by {show_by}")
top_tracks_plays_chart = alt.Chart(top_tracks_df).mark_bar().encode(
    y=alt.X(f'{axis_y}:Q', title=label_y),
    x=alt.Y('trackName:N', title=None, sort='-y', axis=alt.Axis(labelOverlap=False, labelAngle=-45)),  # Sort y-axis by number of plays
    color=alt.value("#FF4B4B"),
)
stat1.altair_chart(top_tracks_plays_chart, use_container_width=True)

top_tracks_xy_df = top_tracks_xy(track_stats_df, top_n)
stat2.subheader(f"Top Tracks by Playtime and Plays")
top_tracks_chart = alt.Chart(top_tracks_xy_df).mark_circle().encode(
    x=alt.X('plays:Q', title='Number of Plays'),
    y=alt.Y('playtime_minutes:Q', title='Total Playtime (minutes)'),
    size=alt.Size('plays:Q', legend=None),
    color=alt.Color('playtime_minutes:Q', legend=None, scale=alt.Scale(scheme='reds')),
    tooltip=['trackName', 'artistName', 'plays', 'playtime_minutes'],
)
stat2.altair_chart(top_tracks_chart, use_container_width=True)