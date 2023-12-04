python

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 128, 0), (128, 255, 0), (0, 128, 255)]
TRACKER_TYPES = {
    'BOOSTING': cv2.TrackerBoosting,
    'MIL': cv2.TrackerMIL,
    'KCF': cv2.TrackerKCF,
    'TLD': cv2.TrackerTLD,
    'MEDIANFLOW': cv2.TrackerMedianFlow,
    'GOTURN': cv2.TrackerGOTURN,
    'MOSSE': cv2.TrackerMOSSE,
}

class LoggedObject:
    DEBUG = False

    def debug(self, *args):
        if self.DEBUG:
            print(*args)

class Settings:
    def __init__(self, config_file=None):
        try:
            with open(config_file) as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            print("Error loading config file:", str(e))
            sys.exit()
        except Exception as e:
            self.config = {}

    @property
    def ALWAYS_FACE_DETECTION(self):
        return not bool(self.config.get("do_tracking", False))

    @property
    def MAX_TRACK_SECONDS(self):
        return self.config.get("max_track_seconds", 0.3)

    @property
    def KEEP_PERSON_SECONDS(self):
        return self.config.get("keep_person_seconds", 5)

    ......

class Person(LoggedObject):
    COLOR_INDEX = 0
    MIN_HR = 70
    MAX_HR = 180

    def __init__(self, cntx, settings):
        self.display_fft = True
        self.cntx = cntx
        self.N = 250
        self.t0 = time.time()
        self.means = []
        self.times = []
        self.magnitude = np.array([])
        self.freqs = np.array([])
        self.color = COLORS[Person.COLOR_INDEX]
        Person.COLOR_INDEX = (Person.COLOR_INDEX + 1) % len(COLORS)
        self.tracker_type = settings.TRACKER_TYPE
        self.font = settings.FONT
        self.last_fd_frame_time = cntx.last_fd_frame_time

    def roi_mean(self):
        global ROI1
        zero = np.zeros(self.cntx.g.shape, np.uint8)
        try:
            ROIONE = ROI1[person_id]
        except:
            ROIONE = ROI1[0]
        cv2.fillConvexPoly(zero, np.array(ROIONE), 1)
        return (zero*self.cntx.g).mean()

    def update_face(self, rectangle=None):
        if rectangle is not None:
            self.last_detection_time = time.time()
            self.rectangle = rectangle
            self.tracker = TRACKER_TYPES[self.tracker_type].create()
            self.tracker.init(self.cntx.g, tuple(self.rectangle))
        self.add_timestamp()

    def track_face(self):
        ok, rectangle = self.tracker.update(self.cntx.g)
        if ok:
            self.last_detection_time = time.time()
            self.rectangle = tuple(map(int, rectangle))
        self.add_timestamp()

    def add_timestamp(self):
        self.times.append(time.time() - self.t0)
        self.times = self.times[-self.N:]
        self.debug("times: %d" % len(self.times))

    def calculate_means(self):
        self.means.append(self.roi_mean())
        self.means = self.means[-self.N:]
        self.debug("means: %d" % len(self.means))

    def calculate_hr(self):
        ......
