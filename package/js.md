# JavaScript
JavaScript是一个轻量级的脚本语言。
是可以插入HTML页面的编程代码。
JavaScript 与 Java 是两种完全不同的语言，无论在概念上还是设计上。
Java（由 Sun 发明）是更复杂的编程语言。
JavaScript 由 Brendan Eich 发明。它于 1995 年出现在 Netscape 中（该浏览器已停止更新），并于 1997 年被 ECMA（一个标准协会）采纳。

HTML中的JavaScript脚本代码必须位于<script>与</script>之间。

脚本代码可以放在<body></body>或者<head></head>之间。

也可以把脚本放在外部文件中，扩展名为.js

JavaScript可以通过不同方式输出数据：

- window.alert()使用弹出框来显示数据
- document.write(),出于测试目的，您可以将JavaScript直接写在HTML 文档中==如果在文档已完成加载后执行 document.write，整个 HTML 页面将被覆盖。==
- innerHTML
- console.log(),如果您的浏览器支持调试，你可以使用 console.log() 方法在浏览器中显示 JavaScript 值。

浏览器中使用 F12 来启用调试模式， 在调试窗口中点击 "Console" 菜单。
- 如需从 JavaScript 访问某个 HTML 元素，您可以使用 document.getElementById(id) 

JavaScript语法：
一般称固定量为字面量

变量用var来定义

==变量是一个名称，字面量是一个值==

注释用//

函数
function functionname(){}

JS对大小写敏感

使用Unicode字符

let 声明的变量只在 let 命令所在的代码块 {} 内有效，在 {} 之外不能访问。

let必须先声明后使用，而var可以先试用后声明

当网页被加载时，浏览器会创建页面的文档对象模型（Document Object Model）

可以在事件发生时执行JavaScript，比如用户在HTML元素上点击时。
如需用户点击某个元素时执行代码，请向一个HTML事件属性添加JavaScript代码:
**onclick=JavaScript**
HTML事件的例子:
当用户点击鼠标时
当网页加载时
当图像加载时
当鼠标移动到元素上时
当输入字段被改变时
当提交HTML表单时
当用户触发按键时

#### html事件属性
当需要向元素分配事件时，可以使用事件属性

#### 使用HTNL DOM来分配事件
document.getElementById("myBtn").onclick=function(){displayDate()};

#### onload与onunload事件
会在用户进入或离开页面时被触发，onload事件可以用于检测访问者的浏览器类型与浏览器版本，并基于这些信息来加载网页的正确版本。
可用于处理cookie

#### onchange事件
常结合对输入字段的验证来使用

#### onmouseover与onmouseout事件
可用于用户的鼠标移至HTML元素上方或者移出元素时触发函数

onmousedown、onmouseup、onclick事件
点击鼠标按钮时会触发onmousedown、释放鼠标按钮时，会触发onmouseup事件、鼠标点击完成时，会触发onclick事件

### DOM LISTTENER
在用户点击按钮时触发监听事件
document.getElementById("myBtn").addEventListener("click", displayDate);

addEvebntListener()方法用于向指定元素添加事件句柄

addEvebntListener()方法添加事件句柄不会覆盖已存在的事件句柄

你可以向一个元素添加多个事件句柄

你可以向同个元素添加多个同类型的事件句柄，如：两个 "click" 事件。

当你使用 addEventListener() 方法时, JavaScript 从 HTML 标记中分离开来，可读性更强， 在没有控制HTML标记时也可以添加事件监听。

你可以使用 removeEventListener() 方法来移除事件的监听。

事件传递有两种方式：
冒泡
先触发内部元素，后触发外部元素

捕获
先触发外部元素，后触发内部元素



