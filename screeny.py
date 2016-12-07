import time
import numpy as np
import wx

app = wx.App()
screen = wx.ScreenDC()

# Used to time screenshot intervals
def timeit(f):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        ret = f(*args, **kwargs)
        print('Seconds elapsed: {}'.format(time.time()-t1))
        return ret
    return wrapper

def screenshot(region=None):
    global screen

    assert type(region) is tuple
    assert len(region) == 4

    # Region is a tuple of (x, y, w, h)
    x = region[0]
    y = region[1]
    w = region[2]
    h = region[3]

    # Construct a bitmap
    bmp = wx.Bitmap(w, h)

    # Fill bitmap delete memory (don't want memory leak)
    mem = wx.MemoryDC(bmp)
    mem.Blit(0, 0, w, h, screen, x, y)
    del mem

    # Convert bitmap to image
    wxB = bmp.ConvertToImage()

    # Get data buffer
    img_data = wxB.GetData()

    # Construct np array from data buffer and reshape it to img
    img_data_str = np.frombuffer(img_data, dtype='uint8')
    img = img_data_str.reshape((h, w, 3))
    return img