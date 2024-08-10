import pandas as pd
import tables  

# code from [https://github.com/tbertinmahieux/MSongsDB]
"""
Thierry Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu

This code contains a set of getters functions to access the fields
from an HDF5 song file (regular file with one song or
aggregate / summary file with many songs)

This is part of the Million Song Dataset project from
LabROSA (Columbia University) and The Echo Nest.


Copyright 2010, Thierry Bertin-Mahieux

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


def open_h5_file_read(h5filename):
    """
    Open an existing H5 file in read mode.
    """
    return tables.open_file(h5filename, mode='r')  


def get_num_songs(h5):
    """
    Return the number of songs contained in this h5 file, i.e. the number of rows
    for all basic informations like name, artist, ...
    """
    return h5.root.metadata.songs.nrows

def get_artist_familiarity(h5,songidx=0):
    """
    Get artist familiarity from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.artist_familiarity[songidx]

def get_artist_hotttnesss(h5,songidx=0):
    """
    Get artist hotttnesss from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.artist_hotttnesss[songidx]

def get_artist_id(h5,songidx=0):
    """
    Get artist id from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.artist_id[songidx]

def get_artist_mbid(h5,songidx=0):
    """
    Get artist musibrainz id from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.artist_mbid[songidx]

def get_artist_playmeid(h5,songidx=0):
    """
    Get artist playme id from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.artist_playmeid[songidx]

def get_artist_7digitalid(h5,songidx=0):
    """
    Get artist 7digital id from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.artist_7digitalid[songidx]

def get_artist_latitude(h5,songidx=0):
    """
    Get artist latitude from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.artist_latitude[songidx]

def get_artist_longitude(h5,songidx=0):
    """
    Get artist longitude from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.artist_longitude[songidx]

def get_artist_location(h5,songidx=0):
    """
    Get artist location from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.artist_location[songidx]

def get_artist_name(h5,songidx=0):
    """
    Get artist name from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.artist_name[songidx]

def get_release(h5,songidx=0):
    """
    Get release from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.release[songidx]

def get_release_7digitalid(h5,songidx=0):
    """
    Get release 7digital id from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.release_7digitalid[songidx]

def get_song_id(h5,songidx=0):
    """
    Get song id from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.song_id[songidx]

def get_song_hotttnesss(h5,songidx=0):
    """
    Get song hotttnesss from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.song_hotttnesss[songidx]

def get_title(h5,songidx=0):
    """
    Get title from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.title[songidx]

def get_track_7digitalid(h5,songidx=0):
    """
    Get track 7digital id from a HDF5 song file, by default the first song in it
    """
    return h5.root.metadata.songs.cols.track_7digitalid[songidx]

def get_similar_artists(h5,songidx=0):
    """
    Get similar artists array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.metadata.songs.nrows == songidx + 1:
        return h5.root.metadata.similar_artists[h5.root.metadata.songs.cols.idx_similar_artists[songidx]:]
    return h5.root.metadata.similar_artists[h5.root.metadata.songs.cols.idx_similar_artists[songidx]:
                                            h5.root.metadata.songs.cols.idx_similar_artists[songidx+1]]

def get_artist_terms(h5,songidx=0):
    """
    Get artist terms array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.metadata.songs.nrows == songidx + 1:
        return h5.root.metadata.artist_terms[h5.root.metadata.songs.cols.idx_artist_terms[songidx]:]
    return h5.root.metadata.artist_terms[h5.root.metadata.songs.cols.idx_artist_terms[songidx]:
                                            h5.root.metadata.songs.cols.idx_artist_terms[songidx+1]]

def get_artist_terms_freq(h5,songidx=0):
    """
    Get artist terms array frequencies. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.metadata.songs.nrows == songidx + 1:
        return h5.root.metadata.artist_terms_freq[h5.root.metadata.songs.cols.idx_artist_terms[songidx]:]
    return h5.root.metadata.artist_terms_freq[h5.root.metadata.songs.cols.idx_artist_terms[songidx]:
                                              h5.root.metadata.songs.cols.idx_artist_terms[songidx+1]]

def get_artist_terms_weight(h5,songidx=0):
    """
    Get artist terms array frequencies. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.metadata.songs.nrows == songidx + 1:
        return h5.root.metadata.artist_terms_weight[h5.root.metadata.songs.cols.idx_artist_terms[songidx]:]
    return h5.root.metadata.artist_terms_weight[h5.root.metadata.songs.cols.idx_artist_terms[songidx]:
                                                h5.root.metadata.songs.cols.idx_artist_terms[songidx+1]]

def get_analysis_sample_rate(h5,songidx=0):
    """
    Get analysis sample rate from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.analysis_sample_rate[songidx]

def get_audio_md5(h5,songidx=0):
    """
    Get audio MD5 from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.audio_md5[songidx]

def get_danceability(h5,songidx=0):
    """
    Get danceability from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.danceability[songidx]

def get_duration(h5,songidx=0):
    """
    Get duration from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.duration[songidx]

def get_end_of_fade_in(h5,songidx=0):
    """
    Get end of fade in from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.end_of_fade_in[songidx]

def get_energy(h5,songidx=0):
    """
    Get energy from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.energy[songidx]

def get_key(h5,songidx=0):
    """
    Get key from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.key[songidx]

def get_key_confidence(h5,songidx=0):
    """
    Get key confidence from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.key_confidence[songidx]

def get_loudness(h5,songidx=0):
    """
    Get loudness from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.loudness[songidx]

def get_mode(h5,songidx=0):
    """
    Get mode from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.mode[songidx]

def get_mode_confidence(h5,songidx=0):
    """
    Get mode confidence from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.mode_confidence[songidx]

def get_start_of_fade_out(h5,songidx=0):
    """
    Get start of fade out from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.start_of_fade_out[songidx]

def get_tempo(h5,songidx=0):
    """
    Get tempo from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.tempo[songidx]

def get_time_signature(h5,songidx=0):
    """
    Get signature from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.time_signature[songidx]

def get_time_signature_confidence(h5,songidx=0):
    """
    Get signature confidence from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.time_signature_confidence[songidx]

def get_track_id(h5,songidx=0):
    """
    Get track id from a HDF5 song file, by default the first song in it
    """
    return h5.root.analysis.songs.cols.track_id[songidx]

def get_segments_start(h5,songidx=0):
    """
    Get segments start array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.segments_start[h5.root.analysis.songs.cols.idx_segments_start[songidx]:]
    return h5.root.analysis.segments_start[h5.root.analysis.songs.cols.idx_segments_start[songidx]:
                                           h5.root.analysis.songs.cols.idx_segments_start[songidx+1]]
    
def get_segments_confidence(h5,songidx=0):
    """
    Get segments confidence array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.segments_confidence[h5.root.analysis.songs.cols.idx_segments_confidence[songidx]:]
    return h5.root.analysis.segments_confidence[h5.root.analysis.songs.cols.idx_segments_confidence[songidx]:
                                                h5.root.analysis.songs.cols.idx_segments_confidence[songidx+1]]

def get_segments_pitches(h5,songidx=0):
    """
    Get segments pitches array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.segments_pitches[h5.root.analysis.songs.cols.idx_segments_pitches[songidx]:,:]
    return h5.root.analysis.segments_pitches[h5.root.analysis.songs.cols.idx_segments_pitches[songidx]:
                                             h5.root.analysis.songs.cols.idx_segments_pitches[songidx+1],:]

def get_segments_timbre(h5,songidx=0):
    """
    Get segments timbre array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.segments_timbre[h5.root.analysis.songs.cols.idx_segments_timbre[songidx]:,:]
    return h5.root.analysis.segments_timbre[h5.root.analysis.songs.cols.idx_segments_timbre[songidx]:
                                            h5.root.analysis.songs.cols.idx_segments_timbre[songidx+1],:]

def get_segments_loudness_max(h5,songidx=0):
    """
    Get segments loudness max array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.segments_loudness_max[h5.root.analysis.songs.cols.idx_segments_loudness_max[songidx]:]
    return h5.root.analysis.segments_loudness_max[h5.root.analysis.songs.cols.idx_segments_loudness_max[songidx]:
                                                  h5.root.analysis.songs.cols.idx_segments_loudness_max[songidx+1]]

def get_segments_loudness_max_time(h5,songidx=0):
    """
    Get segments loudness max time array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.segments_loudness_max_time[h5.root.analysis.songs.cols.idx_segments_loudness_max_time[songidx]:]
    return h5.root.analysis.segments_loudness_max_time[h5.root.analysis.songs.cols.idx_segments_loudness_max_time[songidx]:
                                                       h5.root.analysis.songs.cols.idx_segments_loudness_max_time[songidx+1]]

def get_segments_loudness_start(h5,songidx=0):
    """
    Get segments loudness start array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.segments_loudness_start[h5.root.analysis.songs.cols.idx_segments_loudness_start[songidx]:]
    return h5.root.analysis.segments_loudness_start[h5.root.analysis.songs.cols.idx_segments_loudness_start[songidx]:
                                                    h5.root.analysis.songs.cols.idx_segments_loudness_start[songidx+1]]

def get_sections_start(h5,songidx=0):
    """
    Get sections start array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.sections_start[h5.root.analysis.songs.cols.idx_sections_start[songidx]:]
    return h5.root.analysis.sections_start[h5.root.analysis.songs.cols.idx_sections_start[songidx]:
                                           h5.root.analysis.songs.cols.idx_sections_start[songidx+1]]

def get_sections_confidence(h5,songidx=0):
    """
    Get sections confidence array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.sections_confidence[h5.root.analysis.songs.cols.idx_sections_confidence[songidx]:]
    return h5.root.analysis.sections_confidence[h5.root.analysis.songs.cols.idx_sections_confidence[songidx]:
                                                h5.root.analysis.songs.cols.idx_sections_confidence[songidx+1]]

def get_beats_start(h5,songidx=0):
    """
    Get beats start array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.beats_start[h5.root.analysis.songs.cols.idx_beats_start[songidx]:]
    return h5.root.analysis.beats_start[h5.root.analysis.songs.cols.idx_beats_start[songidx]:
                                        h5.root.analysis.songs.cols.idx_beats_start[songidx+1]]

def get_beats_confidence(h5,songidx=0):
    """
    Get beats confidence array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.beats_confidence[h5.root.analysis.songs.cols.idx_beats_confidence[songidx]:]
    return h5.root.analysis.beats_confidence[h5.root.analysis.songs.cols.idx_beats_confidence[songidx]:
                                             h5.root.analysis.songs.cols.idx_beats_confidence[songidx+1]]

def get_bars_start(h5,songidx=0):
    """
    Get bars start array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.bars_start[h5.root.analysis.songs.cols.idx_bars_start[songidx]:]
    return h5.root.analysis.bars_start[h5.root.analysis.songs.cols.idx_bars_start[songidx]:
                                       h5.root.analysis.songs.cols.idx_bars_start[songidx+1]]

def get_bars_confidence(h5,songidx=0):
    """
    Get bars start array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.bars_confidence[h5.root.analysis.songs.cols.idx_bars_confidence[songidx]:]
    return h5.root.analysis.bars_confidence[h5.root.analysis.songs.cols.idx_bars_confidence[songidx]:
                                            h5.root.analysis.songs.cols.idx_bars_confidence[songidx+1]]

def get_tatums_start(h5,songidx=0):
    """
    Get tatums start array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.tatums_start[h5.root.analysis.songs.cols.idx_tatums_start[songidx]:]
    return h5.root.analysis.tatums_start[h5.root.analysis.songs.cols.idx_tatums_start[songidx]:
                                         h5.root.analysis.songs.cols.idx_tatums_start[songidx+1]]

def get_tatums_confidence(h5,songidx=0):
    """
    Get tatums confidence array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.analysis.songs.nrows == songidx + 1:
        return h5.root.analysis.tatums_confidence[h5.root.analysis.songs.cols.idx_tatums_confidence[songidx]:]
    return h5.root.analysis.tatums_confidence[h5.root.analysis.songs.cols.idx_tatums_confidence[songidx]:
                                              h5.root.analysis.songs.cols.idx_tatums_confidence[songidx+1]]

def get_artist_mbtags(h5,songidx=0):
    """
    Get artist musicbrainz tag array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.musicbrainz.songs.nrows == songidx + 1:
        return h5.root.musicbrainz.artist_mbtags[h5.root.musicbrainz.songs.cols.idx_artist_mbtags[songidx]:]
    return h5.root.musicbrainz.artist_mbtags[h5.root.metadata.songs.cols.idx_artist_mbtags[songidx]:
                                             h5.root.metadata.songs.cols.idx_artist_mbtags[songidx+1]]

def get_artist_mbtags_count(h5,songidx=0):
    """
    Get artist musicbrainz tag count array. Takes care of the proper indexing if we are in aggregate
    file. By default, return the array for the first song in the h5 file.
    To get a regular numpy ndarray, cast the result to: numpy.array( )
    """
    if h5.root.musicbrainz.songs.nrows == songidx + 1:
        return h5.root.musicbrainz.artist_mbtags_count[h5.root.musicbrainz.songs.cols.idx_artist_mbtags[songidx]:]
    return h5.root.musicbrainz.artist_mbtags_count[h5.root.metadata.songs.cols.idx_artist_mbtags[songidx]:
                                                   h5.root.metadata.songs.cols.idx_artist_mbtags[songidx+1]]

def get_year(h5,songidx=0):
    """
    Get release year from a HDF5 song file, by default the first song in it
    """
    return h5.root.musicbrainz.songs.cols.year[songidx]


def extract_song_data(h5):
    num_songs = get_num_songs(h5)
    data = {
        'artist_familiarity': [],
        'artist_hotttnesss': [],
        'artist_id': [],
        'artist_mbid': [],
        'artist_playmeid': [],
        'artist_7digitalid': [],
        'artist_latitude': [],
        'artist_longitude': [],
        'artist_location': [],
        'artist_name': [],
        'release': [],
        'release_7digitalid': [],
        'song_id': [],
        'song_hotttnesss': [],
        'title': [],
        'track_7digitalid': [],
        'similar_artists': [],
        'artist_terms': [],
        'artist_terms_freq': [],
        'artist_terms_weight': [],
        'analysis_sample_rate': [],
        'audio_md5': [],
        'danceability': [],
        'duration': [],
        'end_of_fade_in': [],
        'energy': [],
        'key': [],
        'key_confidence': [],
        'loudness': [],
        'mode': [],
        'mode_confidence': [],
        'start_of_fade_out': [],
        'tempo': [],
        'time_signature': [],
        'time_signature_confidence': [],
        'track_id': [],
        'segments_start': [],
        'segments_confidence': [],
        'segments_pitches': [],
        'segments_timbre': [],
        'segments_loudness_max': [],
        'segments_loudness_max_time': [],
        'segments_loudness_start': [],
        'sections_start': [],
        'sections_confidence': [],
        'beats_start': [],
        'beats_confidence': [],
        'bars_start': [],
        'bars_confidence': [],
        'tatums_start': [],
        'tatums_confidence': [],
        'artist_mbtags': [],
        'artist_mbtags_count': [],
        'year': [],
    }
    
    for i in range(num_songs):
        data['artist_familiarity'].append(get_artist_familiarity(h5, i))
        data['artist_hotttnesss'].append(get_artist_hotttnesss(h5, i))
        data['artist_id'].append(get_artist_id(h5, i))
        data['artist_mbid'].append(get_artist_mbid(h5, i))
        data['artist_playmeid'].append(get_artist_playmeid(h5, i))
        data['artist_7digitalid'].append(get_artist_7digitalid(h5, i))
        data['artist_latitude'].append(get_artist_latitude(h5, i))
        data['artist_longitude'].append(get_artist_longitude(h5, i))
        data['artist_location'].append(get_artist_location(h5, i))
        data['artist_name'].append(get_artist_name(h5, i))
        data['release'].append(get_release(h5, i))
        data['release_7digitalid'].append(get_release_7digitalid(h5, i))
        data['song_id'].append(get_song_id(h5, i))
        data['song_hotttnesss'].append(get_song_hotttnesss(h5, i))
        data['title'].append(get_title(h5, i))
        data['track_7digitalid'].append(get_track_7digitalid(h5, i))
        data['similar_artists'].append(get_similar_artists(h5, i))
        data['artist_terms'].append(get_artist_terms(h5, i))
        data['artist_terms_freq'].append(get_artist_terms_freq(h5, i))
        data['artist_terms_weight'].append(get_artist_terms_weight(h5, i))
        data['analysis_sample_rate'].append(get_analysis_sample_rate(h5, i))
        data['audio_md5'].append(get_audio_md5(h5, i))
        data['danceability'].append(get_danceability(h5, i))
        data['duration'].append(get_duration(h5, i))
        data['end_of_fade_in'].append(get_end_of_fade_in(h5, i))
        data['energy'].append(get_energy(h5, i))
        data['key'].append(get_key(h5, i))
        data['key_confidence'].append(get_key_confidence(h5, i))
        data['loudness'].append(get_loudness(h5, i))
        data['mode'].append(get_mode(h5, i))
        data['mode_confidence'].append(get_mode_confidence(h5, i))
        data['start_of_fade_out'].append(get_start_of_fade_out(h5, i))
        data['tempo'].append(get_tempo(h5, i))
        data['time_signature'].append(get_time_signature(h5, i))
        data['time_signature_confidence'].append(get_time_signature_confidence(h5, i))
        data['track_id'].append(get_track_id(h5, i))
        data['segments_start'].append(get_segments_start(h5, i))
        data['segments_confidence'].append(get_segments_confidence(h5, i))
        data['segments_pitches'].append(get_segments_pitches(h5, i))
        data['segments_timbre'].append(get_segments_timbre(h5, i))
        data['segments_loudness_max'].append(get_segments_loudness_max(h5, i))
        data['segments_loudness_max_time'].append(get_segments_loudness_max_time(h5, i))
        data['segments_loudness_start'].append(get_segments_loudness_start(h5, i))
        data['sections_start'].append(get_sections_start(h5, i))
        data['sections_confidence'].append(get_sections_confidence(h5, i))
        data['beats_start'].append(get_beats_start(h5, i))
        data['beats_confidence'].append(get_beats_confidence(h5, i))
        data['bars_start'].append(get_bars_start(h5, i))
        data['bars_confidence'].append(get_bars_confidence(h5, i))
        data['tatums_start'].append(get_tatums_start(h5, i))
        data['tatums_confidence'].append(get_tatums_confidence(h5, i))
        data['artist_mbtags'].append(get_artist_mbtags(h5, i))
        data['artist_mbtags_count'].append(get_artist_mbtags_count(h5, i))
        data['year'].append(get_year(h5, i))
    
    return data


def h5_to_dataframe(h5_filename):
    with open_h5_file_read(h5_filename) as h5:
        data = extract_song_data(h5)
        df = pd.DataFrame(data)
    return df
import os

def process_all_h5_files(base_directory):
    all_data = []
    for subdir, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.h5'):
                file_path = os.path.join(subdir, file)
                df = h5_to_dataframe(file_path)
                all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


base_directory = r'C:\Users\resha\data\millionsongsubset'
combined_df = process_all_h5_files(base_directory)
# read in every column and set to csv
combined_df.to_csv(r"C:\Users\resha\data\MSD_subset.csv")  
