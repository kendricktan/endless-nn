from pykeyboard import PyKeyboard, PyKeyboardEvent

class ClickKeyEventListener(PyKeyboardEvent):
    def __init__(self, *args, **kwargs):
        self.end = False
        super(ClickKeyEventListener, self).__init__(*args, **kwargs)

    def tap(self, keycode, character, press):  # press is boolean; True for press, False for release
        if character == 'q':
            self.end = True
        else:
            print(character)

k = ClickKeyEventListener()
k.start()

while not k.end:
    pass
