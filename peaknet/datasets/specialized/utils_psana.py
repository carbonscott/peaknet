import numpy as np
import psana

class PsanaImg:
    """
    It serves as an image accessing layer based on the data management system
    psana in LCLS.  
    """

    def __init__(self, exp, run, mode, detector_name):

        # Biolerplate code to access an image
        # Set up data source
        self.datasource_id = f"exp={exp}:run={run}:{mode}"
        self.datasource    = psana.DataSource( self.datasource_id )
        self.run_current   = next(self.datasource.runs())
        self.timestamps    = self.run_current.times()

        # Set up detector
        self.detector = psana.Detector(detector_name)

        # Set image reading mode
        self.read = { "raw"   : self.detector.raw,
                      "calib" : self.detector.calib,
                      "image" : self.detector.image,
                      "mask"  : self.detector.mask, }


    def __len__(self):
        return len(self.timestamps)


    def get(self, event_num, id_panel = None, mode = "calib"):
        # Fetch the timestamp according to event number...
        timestamp = self.timestamps[event_num]

        # Access each event based on timestamp...
        event = self.run_current.event(timestamp)

        # Only two modes are supported...
        assert mode in ("raw", "calib", "image"), \
            f"Mode {mode} is not allowed!!!  Only 'raw', 'calib' and 'image' are supported."

        # Fetch image data based on timestamp from detector...
        data = self.read[mode](event)
        img  = data[int(id_panel)] if id_panel is not None else data

        return img


    def assemble(self, multipanel = None, mode = "image", fake_event_num = 0):
        # Set up a fake event_num...
        event_num = fake_event_num

        # Fetch the timestamp according to event number...
        timestamp = self.timestamps[int(event_num)]

        # Access each event based on timestamp...
        event = self.run_current.event(timestamp)

        # Fetch image data based on timestamp from detector...
        img = self.read[mode](event, multipanel)

        return img


    def create_bad_pixel_mask(self):
        return self.read["mask"](self.run_current, calib       = True,
                                                   status      = True,
                                                   edges       = True,
                                                   central     = True,
                                                   unbond      = True,
                                                   unbondnbrs  = True,
                                                   unbondnbrs8 = False).astype(np.uint16)
