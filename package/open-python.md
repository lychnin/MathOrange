## cv2.imread()
读取图片
格式：cv2.imread(path, flag)

Parameters:
path: 图片地址.
flag: 指定读取的样式，0，-1，1，各代表不同的样式，比如彩图、灰度图、alpha通道图

Return Value: This method returns an image that is loaded from the specified file.
cv2.IMREAD_COLOR: It specifies to load a color image. Any transparency of image will be neglected. It is the default flag. Alternatively, we can pass integer value 1 for this flag.
cv2.IMREAD_GRAYSCALE: It specifies to load an image in grayscale mode. Alternatively, we can pass integer value 0 for this flag.
cv2.IMREAD_UNCHANGED: It specifies to load an image as such including alpha channel. Alternatively, we can pass integer value -1 for this flag.

## cv2.getTextSize(text,fontFace,fontScale,thickness)

text：文本内容，
fontFace：字体,
fontScale：字体大小,
thickness：文本用线的宽度

获得文本框参数，返回文本框长、高以及基线距离文本框的距离
## cv2.putText(image,text,org,fontFace,fontScale,color,thickness,lineType,bottomLeftOrgin)
将文本写入

img：写入文本的图片
text:文本值
org	Bottom-left：文本的左下顶点
fontFace:字体
fontScale：字体大小
color：文本的颜色
thickness：文本用线的宽度
lineType：文本用线的线型
bottomLeftOrigin：决定了左侧顶点位于上方还是下方，True时位于下方，False时位于上方


## cv2.rectangle(img,pt1,pt2,color,thickness,linetype,shift)
img	Image.
pt1	矩形的对角线上的一个顶点
pt2	矩形的对角线上的另一顶点
color 矩形颜色
thickness 矩形线宽
lineType 矩形线型
shift 点坐标中的小数位数
notes:pt1与pt2中的数字需要为整型


## cv2.circle(img,center,radius,color,thickness,linetype,shift)

img	Image where the circle is drawn.
center	Center of the circle.
radius	Radius of the circle.
color	Circle color.
thickness	Thickness of the circle outline, if positive. Negative values, like FILLED, mean that a filled circle is to be drawn.
lineType	Type of the circle boundary. See LineTypes
shift	Number of fractional bits in the coordinates of the center and in the radius value.


