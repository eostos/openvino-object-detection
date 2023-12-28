'''
Created on Mar 13, 2019

@author: ebenezer2
'''

"""Class and methods for rectangle operations"""

class Rect(object):
    """Data type for operations with rectangle regions"""

    def __init__(self):
        self.x_left = 0
        self.y_top = 0
        self.x_right = 10
        self.y_bot = 10
        self.ratio = 1

    def coordinates(self):
        """Return tuple with coordinates of the rectangle object"""
        return (self.x_left, self.y_top), (self.x_right, self.y_bot)

    def get_ratio(self):
        """Return the ratio of the rectangle object"""
        return (self.x_right - self.x_left) / (self.y_bot - self.y_top)

    def rectangle(self, x_left, y_top, height, width):
        """Define rectangle with origin point and size"""
        self.x_left = x_left
        self.y_top = y_top
        self.x_right = self.x_left + width
        self.y_bot = self.y_top + height
        self.ratio = self.get_ratio()
        return self

    def rectangle_p2p(self, x_left, y_top, x_right, y_bottom):
        """Define rectangle with the topleft and bottomright points (point to point)"""
        if x_right > x_left and y_bottom > y_top:
            self.x_left = x_left
            self.y_top = y_top
            self.x_right = x_right
            self.y_bot = y_bottom
            self.ratio = self.get_ratio()
            return self
        else:
            print("Convention error: Upper point is not the smallest point")

    def rectangle_from_yolo_mark(self, x, y, w, h, img_size):
        """Convert Yolo mark coordinates to Rect object"""
        x_left = (x - w / 2.0) * img_size[1]
        y_top = (y - h / 2.0) * img_size[0]
        return self.rectangle(int(x_left), int(y_top), int(h * img_size[0]), int(w * img_size[1]))

    def rectangle_to_yolo_mark(self, img_size):
        """Convert Rect object to Yolo mark coordinates"""
        w = self.x_right - self.x_left
        h = self.y_bot - self.y_top
        x = (self.x_left + w / 2.0) / img_size[1]
        y = (self.y_top + h / 2.0) / img_size[0]
        w = float(w) / img_size[1]
        h = float(h) / img_size[0]
        rectangle = '{x} {y} {w} {h}'.format(x=x, y=y, w=w, h=h)
        return rectangle

    def shift(self, x_shift, y_shift):
        """Shift the Rect object x and y times"""
        x = self.x_left + int(x_shift)
        y = self.y_top + int(y_shift)
        return self.rectangle(x, y, self.y_bot - self.y_top, self.x_right - self.x_left)

    def shrink(self, x0_add, x1_add, y0_add, y1_add):
        """Shrink the Rect object according to the input arguments"""
        self.x_right -= int(abs(x0_add))
        self.x_left += int(abs(x1_add))
        self.y_bot -= int(abs(y0_add))
        self.y_top += int(abs(y1_add))
        return self

    def expand(self, x0_add, x1_add, y0_add, y1_add):
        """Expand the Rect object according to the input arguments"""
        self.x_right += int(abs(x0_add))
        self.x_left -= int(abs(x1_add))
        self.y_bot += int(abs(y0_add))
        self.y_top -= int(abs(y1_add))
        return self
    
    def empalme(self, x0_add, x1_add, y0_add, y1_add):
        """Expand the Rect object according to the input arguments"""
        self.x_right -= int(abs(x0_add))
        self.x_left -= int(abs(x1_add))
        self.y_bot -= int(abs(y0_add))
        self.y_top -= int(abs(y1_add))
        return self
    
    def portion_intersected(self, rectangle2):
        """Calculate the normalized proportion of intersection"""
        area = (self.x_right - self.x_left + 1) * (self.y_bot - self.y_top + 1)
        ix_left = max(self.x_left, rectangle2.x_left)
        iy_top = max(self.y_top, rectangle2.y_top)
        ix_right = min(self.x_right, rectangle2.x_right)
        iy_bot = min(self.y_bot, rectangle2.y_bot)
        dx = ix_right - ix_left + 1
        dy = iy_bot - iy_top + 1
        area_intersection = (dx) * (dy)
        #print(dx,dy,area_intersection)
        return area_intersection / area

    def check_intersection(self, rectangle2):
        """Check if two rectangles are intersected"""
        x_intersect = True
        y_intersect = True
        if (self.x_left > rectangle2.x_right) or (self.x_right < rectangle2.x_left):
            x_intersect = False
        if (self.y_top > rectangle2.y_bot) or (self.y_bot < rectangle2.y_top):
            y_intersect = False
        return x_intersect and y_intersect, self.portion_intersected(rectangle2)

    def merge_rectangles(self, rectangle2):
        """Merge two rectangles"""
        self.x_left = min(self.x_left, rectangle2.x_left)
        self.y_top = min(self.y_top, rectangle2.y_top)
        self.x_right = max(self.x_right, rectangle2.x_right)
        self.y_bot = max(self.y_bot, rectangle2.y_bot)
        self.ratio = self.get_ratio()
        return self

    def crop_image(self, img):
        """Crop image at the region said by the rectangle"""
        if self.x_left < 0:
            self.x_left = 0
        if self.y_top < 0:
            self.y_top = 0
        if self.y_bot > img.shape[0]:
            self.y_bot = img.shape[0]
        if self.x_right > img.shape[1]:
            self.x_right = img.shape[1]
        cropped = img[self.y_top:self.y_bot, self.x_left:self.x_right]
        return cropped