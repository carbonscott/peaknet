#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import regex

class StreamParser:
    '''
    Extract successfully indexed peaks previously found by psocake peak finder.

    Remark: Avoid parsing a stream file if you can.  CrystFEL stream file has
    several levels of ambiguities to handle (:, =, 2 kinds of tabulated data).
    Python's lack of non-backtracking in regex (the '(?>)' block) will
    exacerbate the file parsing.

    '''

    def __init__(self, path_stream):
        self.path_stream = path_stream

        self.marker_dict = {
            "CHUNK_START"      : "----- Begin chunk -----",
            "CHUNK_END"        : "----- End chunk -----",
            "GEOM_START"       : "----- Begin geometry file -----",
            "GEOM_END"         : "----- End geometry file -----",
            "PEAK_LIST_START"  : "Peaks from peak search",
            "PEAK_LIST_END"    : "End of peak list",
            "CRYSTAL_START"    : "--- Begin crystal",
            "CRYSTAL_END"      : "--- End crystal",
            "REFLECTION_START" : "Reflections measured after indexing",
            "REFLECTION_END"   : "End of reflections",
        }

        self.regex_dict = self.init_regex()

        # Keep results in stream_dict...
        self.stream_dict = {}


    def init_regex(self):
        regex_dict = {}

        # Parse peaks found...
        regex_dict['peak_found'] = regex.compile( 
            r"""
            (?x)
            (?>
                (?:
                    (?>
                        (?&FLOAT) # Match a floating number
                    )
                    \s+           # Match whitespace at least once
                ){4}              # Match the whole construct 4 times
            )

            (?&DET_PANEL)         # Match a detector panel

            (?(DEFINE)
                (?<FLOAT>
                    [-+]?         # Match a sign
                    (?>\d+)       # Match integer part
                    (?:\.\d*)?    # Match decimal part
                )
                (?<DET_PANEL>
                    (?: [0-9A-Za-z]+ $ )
                )
            )
            """
        )

        # Parse peaks found...
        regex_dict['peak_indexed'] = regex.compile( 
            r"""
            (?x)
            (?:
                (?>
                    (?&FLOAT) # Match a floating number
                )
                \s+           # Match whitespace at least once
            ){9}              # Match the whole construct 4 times

            (?&DET_PANEL)     # Match a detector panel

            (?(DEFINE)
                (?<FLOAT>
                    [-+]?         # Match a sign
                    (?>\d+)       # Match integer part
                    (?:\.\d*)?    # Match decimal part
                )
                (?<DET_PANEL>
                    (?: [0-9A-Za-z]+ $ )
                )
            )
            """
        )

        # Parse colon item in event...
        # DISCARD!!! Don't match absolute words with regex
        regex_dict['chunk_colon'] = regex.compile(
            r"""
            (?x)
            (?>
                (?P<KEY>     # Match a keyword below
                    (?:Image \s filename)
                |   (?:Event)
                )
            )
            :                # Match a colon
            (?P<VALUE>.+ $)  # Match the event number
            """
        )

        # Parse equal sign item in event...
        regex_dict['chunk_eq'] = regex.compile(
            r"""
            (?x)
            (?>
                (?P<KEY>     # Match a keyword below
                    indexed_by
                )
            )
            \s = \s          # Match a equal sign with blank spaces on both sides
            (?P<VALUE>.+ $)  # Match the event number
            """
        )

        # Parse detector gemoetry...
        regex_dict['geom'] = regex.compile(
            r"""
            (?x)
            # Match the pattern below
            (?> (?&DET_PANEL) / ) (?&COORD)
            \s = \s    # Match a equal sign with blank spaces on both sides
            (?&VALUE)  # Match the value of the coordinate

            (?(DEFINE)
                (?<DET_PANEL>
                    [0-9a-zA-Z]+
                )
                (?<COORD>
                    (?:min_fs)
                |   (?:min_ss)
                |   (?:max_fs)
                |   (?:max_ss)
                )
                (?<VALUE> [0-9]+ $)
            )
            """
        )

        return regex_dict


    def parse(self):
        '''
        Return a dictionary of events that contain successfully indexed peaks
        found by psocake peak finder.
        '''
        # Define state variable for parse/skipping lines...
        is_chunk_found = False
        save_chunk_ok  = False
        save_geom_ok   = True
        in_peaklist    = False
        in_indexedlist = False
        in_geom        = False

        # Get all the regex for string matching...
        regex_dict = self.regex_dict

        # Import marker...
        marker_dict = self.marker_dict

        stream_dict = {
            'geom'         : {},
            'chunk'        : {},
        }

        # Start parsing a stream file by chunks...
        path_stream = self.path_stream
        with open(path_stream,'r') as fh:
            # Go through each line...
            for line in fh:
                # Strip surrounding blank spaces...
                line = line.strip()

                # Okay to save geometry???
                if save_geom_ok:
                    # Match a geom object...
                    m = regex_dict['geom'].match(line)
                    if m is not None:
                        # Fetch values...
                        capture_dict = m.capturesdict()
                        panel = capture_dict['DET_PANEL'][0]
                        coord = capture_dict['COORD'][0]
                        value = capture_dict['VALUE'][0]

                        # Save values...
                        if not panel in stream_dict['geom']:
                            stream_dict['geom'][panel] = {
                                'min_fs' : None,
                                'min_ss' : None,
                                'max_fs' : None,
                                'max_ss' : None,
                            }
                        stream_dict['geom'][panel][coord] = int(value)

                        continue

                # Locate the beginning and end of a new chunk...
                # Consider chunk is found
                if line == marker_dict["CHUNK_START"]: 
                    is_chunk_found = True
                    save_chunk_ok  = False    # ...Don't save any peaks by default
                    save_geom_ok   = False

                # Consider chunk not found at the end of a chunk
                if line == marker_dict["CHUNK_END"]: is_chunk_found = False

                # Skip parsing statements below if a chunk is not found
                if not is_chunk_found: continue


                # Find filename of this chunk...
                # Look up filename
                if line.startswith("Image filename: "):
                    filename = line[line.rfind(':') + 1:] # e.g. 'Image filename: /xxx/cxig3514_0041.cxi'
                    filename = filename.strip()

                # Find event number (only makes sense to crytfel) of this chunk...
                if line.startswith("Event: "):
                    event_num_crystfel = line[line.rfind('/') + 1:] # e.g. 'Event: //17'
                    event_num_crystfel = int(event_num_crystfel)

                # Find indexing status of this chunk...
                if line.startswith("indexed_by"):
                    status_indexing = line[line.rfind('=') + 1:].strip() # e.g. 'indexed_by = none'

                    # Allow to save this chunk if index is sucessful...
                    if status_indexing != 'none': save_chunk_ok = True

                # Don't save this chunk if indexing is not successful...
                if not save_chunk_ok: continue


                # Ready to save peaks in this chunk...
                # Save by filename
                if not filename in stream_dict['chunk']: stream_dict['chunk'][filename] = {}

                # Save by event number
                if not event_num_crystfel in stream_dict['chunk'][filename]:
                    stream_dict['chunk'][filename][event_num_crystfel] = {}

                # Begin a peak list
                if line == marker_dict["PEAK_LIST_START"]:
                    in_peaklist = True
                    is_first_line_in_peaklist = True
                    continue

                # Exit a peak list
                if line == marker_dict["PEAK_LIST_END"]:
                    in_peaklist = False
                    continue

                # Save peaks in a peak list...
                if in_peaklist:
                    # Skip the header
                    if is_first_line_in_peaklist:
                        is_first_line_in_peaklist = False
                        continue

                    # Saving
                    dim1, dim2, _, _, panel = line.split()

                    if not panel in stream_dict['chunk'][filename][event_num_crystfel]: 
                        stream_dict['chunk'           ] \
                                   [filename          ] \
                                   [event_num_crystfel] \
                                   [panel             ] = { 'found' : [], 'indexed' : [] }

                    stream_dict['chunk'           ] \
                               [filename          ] \
                               [event_num_crystfel] \
                               [panel             ] \
                               ['found'           ].append((float(dim1), 
                                                            float(dim2),))
                    continue

                # Find an indexed list
                if line == marker_dict["REFLECTION_START"]:
                    in_indexedlist = True
                    is_first_line_in_indexedlist = True
                    continue

                # Exit an indexed list
                if line == marker_dict["REFLECTION_END"]:
                    in_indexedlist = False
                    continue

                # Save peaks in an indexed list...
                if in_indexedlist:
                    # Skip the header
                    if is_first_line_in_indexedlist:
                        is_first_line_in_indexedlist = False
                        continue

                    # Saving
                    _, _, _, _, _, _, _, dim1, dim2, panel = line.split()

                    # If the panel doesn't have found peak, skip it
                    if not panel in stream_dict['chunk'][filename][event_num_crystfel]: continue

                    # Otherwise, save it
                    stream_dict['chunk'           ] \
                               [filename          ] \
                               [event_num_crystfel] \
                               [panel             ] \
                               ['indexed'         ].append((float(dim1),
                                                            float(dim2),))
                    continue

        self.stream_dict = stream_dict




class GeomInterpreter:

    def __init__(self, geom_dict):
        self.geom_dict = geom_dict


    def to_cheetah(self, offset = 0):
        geom_dict = self.geom_dict

        cheetah_geom_dict = {}
        for panel, coord_dict in geom_dict.items():
            x_min, y_min, x_max, y_max = coord_dict.values()
            x_min += offset
            y_min += offset
            x_max += offset
            y_max += offset
            cheetah_geom_dict[panel] = (x_min, y_min, x_max, y_max)

        return cheetah_geom_dict


    def to_python(self, offset = 0):
        '''
        Cheetah range is closed on the right, but Python range is open.
        '''
        geom_dict = self.geom_dict

        cheetah_geom_dict = {}
        for panel, coord_dict in geom_dict.items():
            x_min, y_min, x_max, y_max = coord_dict.values()

            # Handle offset...
            x_min += offset
            y_min += offset
            x_max += offset
            y_max += offset

            # Include the rightmost coordinate by incrementing one...
            x_max += 1
            y_max += 1

            cheetah_geom_dict[panel] = (x_min, y_min, x_max, y_max)

        return cheetah_geom_dict
