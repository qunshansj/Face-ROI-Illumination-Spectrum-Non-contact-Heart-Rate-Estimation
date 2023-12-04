python

class LoggedObject:
    DEBUG = False  #

    def debug(self, *args):
        if self.DEBUG:
            print(*args)


class Settings:  # setări preluare date din fișier de configurare
    def __init__(self, config_file=None):
        try:
            with open(config_file) as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            print("Error loading config file:", str(e))  # eroare la încărcare
            sys.exit()
        except Exception as e:
            self.config = {}

    @property
    def ALWAYS_FACE_DETECTION(self):
        # detecția se face sau nu
        return not bool(self.config.get("do_tracking", False))

    @property
    def MAX_TRACK_SECONDS(self):
        # câte secunde se face urmărire, fără detecție
        return self.config.get("max_track_seconds", 0.3)

    @property
    def KEEP_PERSON_SECONDS(self):
        # după câte secunde o față devine istorică
        return self.config.get("keep_person_seconds", 5)

    @property
    def CASCADE_FILENAME(self):
        # calea către fișierul cascadă utilizat
        program_dir = os.path.dirname(os.path.abspath(__file__))
        default_cascade = os.path.join(program_dir, 'haarcascade_frontalface_default.xml')
        print(f"XML:{default_cascade}")
        return self.config.get("cascade_filename", default_cascade)

    @property
    def SCALE_FACTOR(self):
        # parametru detector - factor de scalare
        return self.config.get("scale_factor", 1.3)

    @property
    def MIN_NEIGHBORS(self):
        # parametru detector - nr minim vecini
        return self.config.get("min_neighbors", 4)

    @property
    def MIN_SIZE_X(self):
        # parametru detector - x minim
        return self.config.get("min_size_x", 50)

    @property
    def MIN_SIZE_Y(self):
        # parametru detector - y minim
        return self.config.get("min_size_y", 50)

    @property
    def TRACKER_TYPE(self):
        # algoritm urmărire prestabilit
        tracker_type = self.config.get("tracker_type", "MIL")
        if tracker_type not in TRACKER_TYPES:
            print("[ERROR]: Invalid tracker_type: %s" % tracker_type)
            tracker_type = "MIL"
        return tracker_type

    @property
    def DETECT_HEARTRATE(self):
        # calculează puls sau nu
        return self.config.get("detect_heartrate", True)

    @property
    def FONT(self):
        return cv2.FONT_HERSHEY_SIMPLEX


class Person(LoggedObject):
    COLOR_INDEX = 0

    MIN_HR = 70
    MAX_HR = 180

    def __init__(self, cntx, settings):
        self.display_fft = True
        self.cntx = cntx
        self.N = 250
        self.t0 = time.time()  # moment de start
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
        #print(zero*self.cntx.g)

        return (zero*self.cntx.g).mean()

    def update_face(self, rectangle=None):  # pe fiecare frame de fd recreez tracker si il initializez
        if rectangle is not None:
            self.last_detection_time = time.time()
            self.rectangle = rectangle  # save rect pt a sti ca este al pers resp
            self.tracker = TRACKER_TYPES[self.tracker_type].create()
            self.tracker.init(self.cntx.g, tuple(self.rectangle))
        self.add_timestamp()

    def track_face(self):
        ok, rectangle = self.tracker.update(self.cntx.g)
        if ok:  # daca tracking se face cu succes
            self.last_detection_time = time.time()  # salveaza momentul de timp
            self.rectangle = tuple(map(int, rectangle))
        self.add_timestamp()

    def add_timestamp(self):
        self.times.append(time.time() - self.t0)
        self.times = self.times[-self.N:]
        self.debug("times: %d" % len(self.times))

    def calculate_means(self):
        self.means.append(self.roi_mean())  # formez un sir din ultima parte din vector si adaug un element
        self.means = self.means[-self.N:]
        self.debug("means: %d" % len(self.means))

    def calculate_hr(self):
        self.calculate_means()
        print(f"Caculated means:{self.means}")
        if len(self.means) < 10:
            return
        y = np.array(self.means, dtype=float)
        n = len(y)  # length of the signal
        fps = float(n) / (self.times[-1] - self.times[0])
        even_times = np.linspace(self.times[0], self.times[-1], n)
        y = np.interp(even_times, self.times, y)
        y = np.hamming(n) * y  # corelatie
        y = y - np.mean(y)
        raw = np.fft.rfft(y * 2)
        fft = np.abs(raw)
        freqs = float(fps) / n * np.arange(n / 2 + 1)
        freqs = 60. * freqs
        print(f"FFT size:{len(fft)}")
        print(f"Freq size:{len(freqs)}")
        idx = np.where((freqs > Person.MIN_HR) & (freqs < Person.MAX_HR))
        self.freqs = freqs[idx]
        self.magnitude = fft[idx]

    @property
    def heart_rate(self):
        print(f"Magnitude:{self.magnitude}")
        if len(self.magnitude) < 10:
            return None
        max_idx = np.argmax(self.magnitude)
        return self.freqs[max_idx]

    def draw_widgets(self):
        x, y, w, h = self.rectangle  # dreptunghiul de față
        cv2.rectangle(self.cntx.frame, (x, y), (x + w, y + h), self.color, 2)
        print(f"Heart rate:{self.heart_rate}")
        if self.heart_rate:
            cv2.putText(self.cntx.frame, "HR:" + str(int(self.heart_rate)), (x + w - 80, y - 4), self.font, 0.8,
                        self.color, 1, cv2.LINE_AA)
        .......
