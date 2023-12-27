from manim import *
import numpy as np
import random
import math

anispeed = 0.5
changeColor = RED

class Thumbnail(MovingCameraScene):
    def construct(self):
        circle = Circle()  # create a circle
        circles = []
        circles.append(circle)
        layer2 = []
        layer3 = []
        layer4 = []
        output = Circle()
        numCircs = 2
        numHiddens = 5
        for x in range(numCircs):
            circles.append(Circle())
        prevCirc = circle
        
        for circ in circles:
            circ.set_color(WHITE)
        
        makeLayer(self, layer2, numHiddens, 8)
        makeLayer(self, layer3, numHiddens, 0)
        makeLayer(self, layer4, numHiddens, -8)
        group = VGroup()
        for item in circles:
            group = VGroup(item, group)
        for item in layer2:
            group = VGroup(item, group)
        for item in layer3:
            group = VGroup(item, group)
        for item in layer4:
            group = VGroup(item, group)
        
        group = VGroup(output, group)
        
        text = Text("YOU Can Make THIS AI", font_size = 180, color = WHITE)

        text.shift(UP*10)

        output.shift(LEFT*-17)
        output.set_color(WHITE)
        
        self.camera.frame.save_state()
        self.play(Create(circle), run_time = 1)  # show the square on screen
        
        self.play(self.camera.frame.animate.set(width = circle.width*10), circle.animate.shift(DOWN*2.5), run_time = anispeed)
        #self.play(circle.animate.shift(DOWN*5 + LEFT*17), run_time = anispeed)
        
        prevCirc = circle
        for i, circ in enumerate(circles): 
            if(circ != prevCirc):
                circ.next_to(prevCirc, UP, buff = 0.5)
            prevCirc = circ

        for circ in circles:
            if circ != circle:
                circ.set_color(changeColor)
                self.play(Create(circ), run_time = anispeed)
                self.play(circ.animate.set_color(WHITE), run_time = anispeed)
        self.wait(.6)
        group1 = moveGroup(circles, 17)
        self.play(self.camera.frame.animate.set(width = circle.width*25), group1, run_time = 1)
        lines12 = connectLayers(circles, layer2)
        lines23 = connectLayers(layer2, layer3)
        lines34 = connectLayers(layer3, layer4)
        lines45 = connectLayers(layer4, output)

        
        
        drawArr(self, layer2, changeColor, anispeed*.9)
        drawArr(self, layer3, changeColor, anispeed*.9)
        drawArr(self, layer4, changeColor, anispeed*.9)
        drawArr(self, [output], changeColor, anispeed*.9)
        drawArr(self, lines12, changeColor, anispeed*.9)
        drawArr(self, lines23, changeColor, anispeed*.9)
        drawArr(self, lines34, changeColor, anispeed*.9)
        drawArr(self, lines45, changeColor, anispeed*.9)
        group.set_stroke(width = 12.0)
        group.set_fill(RED, 1)
        self.play(group.animate.set_color_by_gradient(GREEN, TEAL))
        
        self.play(Create(text))
        self.wait(5)


class Six(MovingCameraScene):
    #Start 3:33
    def construct(self):
        self.camera.frame.set(width = 30)
        vW = MathTex(r"\begin{bmatrix} w_{00} \ w_{10} \ w_{20} \ w_{30} \ w_{40} \\  w_{01} \ w_{11} \ w_{21} \ w_{31} \ w_{41} \\ w_{02} \ w_{12} \ w_{22} \ w_{32} \ w_{42} \\ w_{03} \ w_{13} \ w_{23} \ w_{33} \ w_{43} \\ w_{04} \ w_{14} \ w_{24} \ w_{34} \ w_{44}\end{bmatrix}", opacity = 1, color = WHITE, font_size = 120)
        vN = MathTex(r"\begin{bmatrix} i_{0} \ i_{1} \ i_{02} \ i_{3} \ i_{4}  \end{bmatrix}", opacity = 1, color = WHITE, font_size = 120)
        vectors = MathTex(r"\begin{bmatrix} w_{00} \\ w_{01} \\ w_{02} \\ w_{03} \\ w_{04} \end{bmatrix} ", r"\begin{bmatrix} w_{10} \\ w_{11} \\ w_{12} \\ w_{13} \\ w_{14} \end{bmatrix}  ", r"\begin{bmatrix} w_{20} \\ w_{21} \\ w_{22} \\ w_{23} \\ w_{24} \end{bmatrix} ", r" \begin{bmatrix} w_{30} \\ w_{31} \\ w_{32} \\ w_{33} \\ w_{34} \end{bmatrix} ", r"\begin{bmatrix} w_{40} \\ w_{41} \\ w_{42} \\ w_{43} \\ w_{44} \end{bmatrix}", opacity = 1, color = WHITE, font_size = 90)
        
        dot = MathTex(r"\cdot", opacity = 1, color = WHITE, font_size = 300)
        vW.shift(RIGHT*6+DOWN*-1)
        dot.move_to(vW)
        dot.shift(LEFT*7.5 + UP*2)
        vN.move_to(dot)
        vN.shift(LEFT*5.4)
        self.play(Write(vW))
        self.play(Write(vN))
        self.play(Write(dot))
        self.wait(1)
        vectors.move_to(vW.get_left(), LEFT)
        self.play(Transform(vW, vectors))
        self.wait(1)
        disclaimer = Text("This is not proper syntax for multiplying one vector by others (or at least nothing I've seen before)").shift(DOWN*8)
        self.play(Write(disclaimer))
        self.wait(1)
        self.play(Unwrite(disclaimer), run_time = 1)
        self.play(Unwrite(vN), Unwrite(vW), Unwrite(dot), run_time = 1)

class Test(MovingCameraScene):
    def construct(self):
        self.camera.frame.set(width = 50)
        numbers = [0.39, 1.03, 0.84, 0.12, 0.57]
        n = makeArrModel(numbers)
        self.play(Create(n))
        self.wait(2)

class Five(MovingCameraScene):
    #This scene starts at 2:26/2:27
    def construct(self):
        self.camera.frame.set(width = 50)
        title = Text("Neural Network Outputs VS Actual Function", font_size = 100)
        title.shift(UP*10)
        self.play(Write(title))
        axes = Axes(x_range = [-5, 5, 1], y_range = [-5, 5, 1], x_length = 10, y_length = 10, axis_config={"include_tip": False}).add_coordinates()
        axes.shift(DOWN*4)
        labels = axes.get_axis_labels(x_label = "x", y_label = "f(x)")
        coords = VGroup(labels, axes)
        coords.shift(UP*2, LEFT*10)
        self.play(DrawBorderThenFill(axes), Write(labels))
        key1 = Text("Green: Actual output of a function the AI is learning", t2c= {"Green": GREEN})
        key2 = Text("Red: A linear activation function based networks guess", t2c= {"Red": RED})
        key3 = Text("Blue: A non linear activation function based networks guess", t2c= {"Blue": BLUE_C})
        key2.move_to(key1.get_left(), LEFT)
        key3.move_to(key1.get_left(), LEFT)
        key2.shift(DOWN*2)
        key3.shift(DOWN*4)
        keys = VGroup(key1, key2, key3)
        keys.shift(UP*3 + RIGHT*10)
        self.play(Write(key1))
        self.play(Write(key2))
        self.play(Write(key3))
        ymxb = MathTex(r"y=mx+b", font_size = 120, color = GREEN)
        ymxb.shift(coords.get_top()+UP)
        self.play(Write(ymxb))
        ymxbLine = Line(coords.get_left()*RIGHT + coords.get_bottom()*UP, coords.get_right()*RIGHT + coords.get_top()*UP).set_color(GREEN)
        self.play(Write(ymxbLine))
        adds = MathTex(r"(1+1+1)+1=3+1", font_size = 100, color = PINK)
        mults = MathTex(r"(2*2*2)*1=8*1", font_size = 100, color = PINK)
        adds.shift(DOWN*6)
        mults.shift(DOWN*6)
        self.wait(1)
        self.play(Write(adds))
        self.wait(1)
        self.play(Transform(adds, mults))
        self.wait(1)
        self.play(Unwrite(adds))
        remember = Text("Think back to a minute ago, all the neuron did was multiply by weights and add a bias", font_size=50).move_to(title)
        think = Text("That means that so far we've taken x values and just multiplied and added them", font_size=50).move_to(title)
        see = Text("That is mathematically y = mx + b. This means the network will struggle with non linear approximations", font_size=50).move_to(title)
        newTitle = Text("Neural Network Outputs VS Actual Function", font_size = 100).move_to(title)
        self.play(Transform(title, remember))
        self.wait(2)
        self.play(Transform(title, think))
        self.wait(2)
        self.play(Transform(title, see))
        self.wait(2)
        self.play(Transform(title, newTitle))
        self.wait(2)
        linOutput = Line(coords.get_left()*RIGHT + coords.get_bottom()*UP, coords.get_right()*RIGHT + coords.get_top()*UP).set_color(RED)
        self.play(Write(linOutput))
        self.wait(2)
        relu = Text("Rectified Linear Unit (ReLU): In python output = max(0, x), in English, if x is greater than 0, return x, else return 0", t2c= {"Rectified Linear Unit (ReLU)": PINK}).shift(DOWN*9)
        self.play(Write(relu), run_time = 3)
        self.wait(2)
        self.play(Unwrite(relu))
        reluText = Text("ReLU Neural Network On Functions").move_to(ymxb)
        disclaimer = Text("Animations Are Not Actual Outputs Just Generalizations Of What Tends To Happen").move_to(coords.get_bottom() + DOWN)
        self.play(Transform(ymxb, reluText))
        self.play(Write(disclaimer))
        reluYMXB = Line(coords.get_left()*RIGHT + coords.get_bottom()*UP, coords.get_right()*RIGHT + coords.get_top()*UP).set_color(BLUE_C)
        graph = axes.plot(lambda x : x*x, color = GREEN, x_range = [-2.3, 2.3])
        graph2 = axes.plot(lambda x : x*x, color = BLUE_C, x_range = [-2.3, 2.3])
        graph3 = axes.plot(lambda x : x*1.8, color = RED, x_range = [-2.9, 2.9])
        self.play(Write(reluYMXB))
        self.play(Unwrite(disclaimer))
        self.play(Transform(ymxbLine, graph))
        self.play(Transform(reluYMXB, graph2))
        self.play(Transform(linOutput, graph3))
        graph = axes.plot(lambda x : math.sin(x), color = GREEN, x_range = [-5, 5])
        graph2 = axes.plot(lambda x : math.sin(x), color = BLUE_C, x_range = [-5, 5])
        graph3 = axes.plot(lambda x : x*1/5, color = RED, x_range = [-5, 5])
        self.play(Transform(ymxbLine, graph))
        self.play(Transform(reluYMXB, graph2))
        self.play(Transform(linOutput, graph3))
        graph = axes.plot(lambda x : math.e**x, color = GREEN, x_range = [-5, 1.8])
        graph2 = axes.plot(lambda x : math.e**x, color = BLUE_C, x_range = [-5, 1.8])
        graph3 = axes.plot(lambda x : 5/4+x*1/4, color = RED, x_range = [-5, 5])
        self.play(Transform(ymxbLine, graph))
        self.play(Transform(reluYMXB, graph2))
        self.play(Transform(linOutput, graph3))
        text = Text("See how a linear functions don't perform?").move_to(ymxb)
        self.play(Transform(ymxb, text))
        self.wait(1)
        self.play(self.camera.frame.animate.set(width = 2000))
        #Scene Close

        
        



class Four(MovingCameraScene):
    def construct(self):
        self.camera.frame.set(width = 50)
        neuron = Text("Inside A Neurons \"Thoughts\"", font_size = 100)
        neuron.shift(UP*10)
        disclaimer = Text("Keep in mind neurons are rarely stored as neuron objects and this is just how we visualize them to make sense")
        disclaimer.shift(UP*10)
        neuron2 = Text("Inside A Neurons \"Thoughts\"", font_size = 100)
        neuron2.shift(UP*10)
        later = Text("I will explain this later, I just needed to delay the animation 3 seconds, so that's what this text is for")
        later.shift(UP*10)
        self.play(Create(neuron))
        self.wait(0.5)
        self.play(Transform(neuron, disclaimer))
        self.wait(1)
        self.play(Transform(neuron, later))
        self.wait(1)
        self.play(Transform(neuron, neuron2))
        numbers = [0.39, 1.03, 0.84, 0.12, 0.57]
        weights = [1, 3.4, 0.13, 8, 0.45]
        nums = arrToTextArr(numbers)
        installNumbers(self, nums)
        ways = arrToTextArr(weights)
        
        installWeights(self, ways, nums)
        t, v = multiplyNums(self, ways, nums, weights, numbers)
        solution = Text(str(sum(v)), color = WHITE)
        solution.move_to(ways[2])
        solution.shift(RIGHT*7)
        self.play(transArr(ways + nums, solution), transArr(t, solution))
        self.wait(0.5)
        self.play(delArr(nums + ways), delArr(t))
        vW = MathTex(r"\begin{bmatrix}" + str(weights[0]) + r"\\" + str(weights[1]) + r"\\"+ str(weights[2]) + r"\\"+ str(weights[3]) + r"\\"+ str(weights[4]) + r"\end{bmatrix}", opacity = 1, color = WHITE, font_size = 120)
        vN = MathTex(r"\begin{bmatrix}" + str(numbers[0]) + r"\ " + str(numbers[1]) + r"\ "+ str(numbers[2]) + r"\ "+ str(numbers[3]) + r"\ "+ str(numbers[4]) + r"\end{bmatrix}", opacity = 1, color = WHITE, font_size = 120)
        dot = MathTex(r"\cdot", opacity = 1, color = WHITE, font_size = 300)
        vW.shift(RIGHT*6+DOWN*2)
        dot.move_to(vW)
        dot.shift(LEFT*3.3 + UP*2)
        vN.move_to(dot)
        vN.shift(LEFT*8)
        self.play(Create(vN))
        self.play(Create(vW))
        self.play(Create(dot))
        self.wait(2)
        numbersArr, adafs = makeArrModel(numbers)
        numbersArr.shift(UP*3 + LEFT*5)
        weightsArr, adsfad = makeArrModel(weights)
        weightsArr.shift(LEFT*5)
        self.play(Transform(vN, numbersArr), Transform(vW, weightsArr), dot.animate.shift(LEFT*9.7))
        line = Line(weightsArr.get_left() + DOWN*2 +LEFT*2, weightsArr.get_right() + DOWN*2  + RIGHT * 2)
        self.play(Create(line))
        ansArr, ansArrPos = makeArrModel(v)
        ansArr.shift(DOWN*4 + LEFT*5)
        self.play(Create(ansArr))
        self.wait(1)
        rbox, strv = foldToVal(self, ansArrPos, v)
        self.play(Unwrite(vN))
        self.play(Unwrite(vW), Uncreate(dot))
        self.play(Uncreate(line))
        self.play(rbox, strv.animate.shift(LEFT*5 + UP*3))
        bias = Text("Bias", color = WHITE, font_size = 60)
        bias.move_to(strv)
        bias.shift(RIGHT*4)
        self.play(Write(bias))
        add = Text("+", color = WHITE, font_size = 90)
        add.move_to(strv)
        add.shift(RIGHT*2)
        self.play(Write(add))
        num = Text("3.28", color = WHITE, font_size = 60)
        num.move_to(bias)
        self.play(Transform(bias, num))
        self.wait(1)
        sol = Text("8.49", color = WHITE, font_size = 60)
        sol.move_to(add)
        self.play(Transform(strv, sol), Transform(bias, sol), Uncreate(add))
        self.play(Uncreate(strv), run_time = 0.001)
        self.play(bias.animate.set_color(GREEN))
        self.play(bias.animate.set_font_size(240))
        
        #Activation Functions 2:14
        activate1 = Text("Activation Function", color = GREEN, font_size = 100).move_to(neuron)
        activate2 = Text("Activation Functions tend to be separate objects from where neurons, as we've seen so far, are stored.").move_to(neuron)
        activate3 = Text("Activation Function", color = WHITE, font_size = 100).move_to(neuron)
        self.play(Transform(neuron, activate1))
        self.wait(1)
        self.play(Transform(neuron, activate2))
        self.wait(1)
        self.play(Transform(neuron, activate3))
        self.play(bias.animate.shift(LEFT*3))
        mult = MathTex(r"\cdot", font_size = 200)
        mult.move_to(bias.get_right()+RIGHT)
        self.play(Write(mult))
        c = Text("C", font_size = 240)
        c.move_to(bias.get_right()+3*RIGHT)
        self.play(Write(c))
        self.wait(1)
        self.play(Unwrite(c), Unwrite(mult), Unwrite(bias), Unwrite(neuron))
        self.wait(1)
def foldToVal(self, posArr, vals):
    runningVal = 0
    prevPos = VGroup()
    for i, pos in enumerate(posArr):
        runningVal += vals[i]
        val = Text(str(runningVal)[:4], font_size = 60)
        box = Square(2)
        val.move_to(box)
        newPos = VGroup(val, box)
        newPos.move_to(pos)
        if i != 0:
            self.play(Transform(prevPos, newPos), Transform(pos, newPos))
            self.play(Uncreate(prevPos),  run_time = .001)
        prevPos = pos
    self.wait(1)
    return AnimationGroup(Uncreate(pos)), val
        
def makeArrModel(arr):
    group = VGroup()
    posA = []
    for i, num in enumerate(arr):
        val = Text(str(num)[:4], font_size = 60)
        box = Square(2)
        val.move_to(box)
        pos = VGroup(box, val)
        if i == 0: 
            group = VGroup(pos, group)
            posA.append(pos)
        else:
            pos.move_to(group.get_right(), LEFT)
            posA.append(pos)
            group = VGroup(group, pos)
        
    return group, posA

def transArr(fromA, toO):
    group = AnimationGroup()
    for o in fromA:
        group = AnimationGroup(group, Transform(o, toO))
    return group

def delArr(arr):
    group = AnimationGroup()
    for val in arr:
        group = AnimationGroup(group, Uncreate(val))
    return group

def multiplyNums(obj, ways, nums, weights, numbers):
    arr = multArr(weights, numbers)
    text = arrToTextArr(arr)
    movArrToArr(text, ways)
    group = AnimationGroup()
    for way, num, text in zip(ways, nums, text):
        group = AnimationGroup(group, num.animate.move_to(way), Transform(num, text), Transform(way, text))
    obj.play(group)
    return text, arr

def movArrToArr(arrM, arrD):
    for m, d in zip(arrM, arrD):
        m.move_to(d)

def multArr(arr1, arr2):
    arr3 = []
    for n, m in zip(arr1, arr2):
        arr3.append(n*m)
    return arr3
def installWeights(obj, weights, numbers):
    group = AnimationGroup()
    for way, num in zip(weights, numbers):
        way.move_to(num)
        way.shift(RIGHT*7, UP*4)
        group = AnimationGroup(Create(way), way.animate.shift(DOWN*4), group)
    obj.play(group)

def arrToTextArr(arr):
    text = []
    for elem in arr:
        text.append(Text(str(elem), color=WHITE))
    return text

def installNumbers(obj, numbers):
    for i, number in enumerate(numbers):
        number.shift(LEFT*26 + UP*(4*(-i+2)))
        obj.play(Create(number), number.animate.shift(RIGHT*8))
        

class Two(MovingCameraScene):
    def construct(self):
        
        self.camera.frame.set(width = 2000)
        back = NumberPlane(background_line_style={"stroke_color": TEAL, "stroke_width": 4, "stroke_opacity": .6}, x_range=[-50, 50, 2], y_range=[-50, 50, 2])
        self.add(back)
        self.play(self.camera.frame.animate.set(width = 50))
        dotsR = makeGpoints(40, RED, -8, 8, 2, .6)
        dotsG = makeGpoints(40, GREEN, -10, 10, 2, -.3)
        drawArr(self, dotsR+dotsG, GREEN, 1)
        f = MathTex(r"f(x, y) = RED | GREEN", font_size = 100, z_index = 1)
        f.move_to(UP*10 + LEFT*15)
        self.play(Create(f))
        newShift = AnimationGroup(wholeArrayColorSwitch(GREEN, dotsG), wholeArrayColorSwitch(RED, dotsR))
        self.play(newShift)
        self.wait(1)
        sidelen = 27
        q1 = Square(sidelen)
        q1.move_to(2*UP, UP+RIGHT)
        q1.set_fill(GREEN, .5)
        q2 = Square(sidelen)
        q2.move_to(2*UP, DOWN+RIGHT)
        q2.set_fill(RED, .5)
        q3 = Square(sidelen)
        q3.move_to(2*UP, DOWN+LEFT)
        q3.set_fill(GREEN, .5)
        q4 = Square(sidelen)
        q4.move_to(2*UP, UP+LEFT)
        q4.set_fill(RED, .5)
        self.play(Create(q1), Create(q2), Create(q3), Create(q4), run_time = 3)
        disclaimer = Text(r"Dot colors represent actual outcomes, boxes represent the neural netowrks prediction for each area.", font_size = 50)
        disclaimer.shift(DOWN*13 + RIGHT*9)
        self.play(Create(disclaimer))
        self.wait(1)
        self.play(self.camera.frame.animate.set(width = 2000), run_time = 1)
        
def makeGpoints(count, color, xfact, yfact, offset, m):
    dots = []
    for i in range(0, count):
        x = (random.random()-.5)*xfact
        y = m*random.random()*x*yfact+offset
        dots.append(Dot(color=color).move_to(UP*y + LEFT*x))
    return dots
        
class One(MovingCameraScene):
    def construct(self):
        circle = Circle()  # create a circle
        circles = []
        circles.append(circle)
        layer2 = []
        layer3 = []
        layer4 = []
        output = Circle()
        numCircs = 2
        numHiddens = 5
        for x in range(numCircs):
            circles.append(Circle())
        prevCirc = circle
        
        for circ in circles:
            circ.set_color(WHITE)
        
        makeLayer(self, layer2, numHiddens, 8)
        makeLayer(self, layer3, numHiddens, 0)
        makeLayer(self, layer4, numHiddens, -8)
        
        text = MathTex(r"f(",r"x_{1}",r",",r" x_{2}", r",", r"x_{3}", r") = y", font_size = 180)

        text.shift(UP*11)

        output.shift(LEFT*-17)
        output.set_color(WHITE)
        
        self.camera.frame.save_state()
        self.play(Create(circle), run_time = 1)  # show the square on screen
        
        self.play(self.camera.frame.animate.set(width = circle.width*10), circle.animate.shift(DOWN*2.5), run_time = anispeed)
        #self.play(circle.animate.shift(DOWN*5 + LEFT*17), run_time = anispeed)
        
        prevCirc = circle
        for i, circ in enumerate(circles): 
            if(circ != prevCirc):
                circ.next_to(prevCirc, UP, buff = 0.5)
            prevCirc = circ

        for circ in circles:
            if circ != circle:
                circ.set_color(changeColor)
                self.play(Create(circ), run_time = anispeed)
                self.play(circ.animate.set_color(WHITE), run_time = anispeed)
        self.wait(.6)
        group = moveGroup(circles, 17)
        self.play(self.camera.frame.animate.set(width = circle.width*25), group, run_time = 1)
        lines12 = connectLayers(circles, layer2)
        lines23 = connectLayers(layer2, layer3)
        lines34 = connectLayers(layer3, layer4)
        lines45 = connectLayers(layer4, output)

        
        
        drawArr(self, layer2, changeColor, anispeed*.9)
        drawArr(self, layer3, changeColor, anispeed*.9)
        drawArr(self, layer4, changeColor, anispeed*.9)
        drawArr(self, [output], changeColor, anispeed*.9)
        drawArr(self, lines12, changeColor, anispeed*.9)
        drawArr(self, lines23, changeColor, anispeed*.9)
        drawArr(self, lines34, changeColor, anispeed*.9)
        drawArr(self, lines45, changeColor, anispeed*.9)
        
        
        self.play(Create(text))
        self.wait(1)
        

        layerBox = Rectangle(YELLOW, circle.height*3 + 1.5, circle.width+1)
        layerBox.move_to(circles[1])
        self.play(Create(layerBox))
        inp =  Text("Input Layer", color = YELLOW)
        inp.move_to(circles[1])
        inp.shift(UP*circle.width*2.5)
        self.play(Create(inp))
        xvals = addXes(self, circles)
        self.wait(1)
        
        outputBox = Rectangle(YELLOW, circle.height + 1, circle.width + 1)
        outputBox.move_to(output)
        self.play(Create(outputBox))
        out =  Text("Output Layer", color = YELLOW)
        out.move_to(output)
        out.shift(UP*circle.width*1.5)
        self.play(Create(out))
        y = MathTex(r"y", font_size= 120)
        y.move_to(output)
        self.play(Create(y))
        self.wait(1)
        self.play(self.camera.frame.animate.set(width = circle.width*.5), run_time = 1)




        self.wait(12)
        #Scene Three
        self.play(self.camera.frame.animate.set(width = circle.width*25))
        self.play(Uncreate(layerBox), Uncreate(inp))
        self.play(self.camera.frame.animate.move_to(output).set(width = output.width*8))
        lin = Text("Linear Regression", color = WHITE)
        log = Text("Logistic Regression", color = WHITE)
        lin.shift(RIGHT*17, DOWN*2)
        log.shift(RIGHT*17, DOWN*3)
        self.play(Create(lin), Create(log))
        self.wait(1)
        out2 = Circle()
        out2.move_to(output)
        out2.set_color(WHITE)
        self.play(Create(out2), run_time = .05)
        deLines = uncreateGroup(lines45)
        self.play(Uncreate(lin), Uncreate(outputBox), Uncreate(out), Uncreate(y), deLines, log.animate.shift(UP*6.3), out2.animate.shift(DOWN*1.3), output.animate.shift(UP*1.3))
        linesOut2 = connectLayers(layer4, [out2, output])
        drawArr(self, linesOut2, RED, .5)
        red = Text("y = RED", color = RED, font_size = 20)
        green = Text("y = GREEN", color = GREEN, font_size = 20)
        red.move_to(out2)
        green.move_to(output)
        lowNum = Text("0.2", color = RED, font_size = 30)
        highNum = Text("0.5", color = GREEN, font_size = 30)
        lowNum.move_to(red)
        highNum.move_to(green)
        self.play(Create(red), Create(green))
        self.play(Transform(red, lowNum), Transform(green, highNum))
        self.play(output.animate.set_fill(GREEN, opacity = 0.5))
        self.play(green.animate.shift(RIGHT*3 + DOWN*1.2))
        spamno = [Text("Spam", color = WHITE, font_size = 25), Text("Not Spam", color = WHITE, font_size = 25)]
        spamno[0].move_to(output)
        spamno[1].move_to(out2)
        self.wait(0.5)
        self.play(Transform(green, spamno[0]), Transform(red, spamno[1]), output.animate.set_fill(GREEN, opacity=0.))
        girlno = [Text("Talk", color = WHITE, font_size = 25), Text("No Talk", color = WHITE, font_size = 25)]
        girlno[0].move_to(output)
        girlno[1].move_to(out2)
        self.play(Transform(green, girlno[0]), Transform(red, girlno[1]))
        self.play(out2.animate.set_fill(RED, opacity = 0.5))
        self.play(red.animate.shift(UP*1.2, RIGHT*3))

        #start linear Regression
        self.play(out2.animate.set_fill(RED, opacity = 0.), Uncreate(red), Uncreate(green))   
        lin2 = Text("Linear Regression", color = WHITE)
        lin2.move_to(log)
        self.play(Transform(log, lin2))
        self.play(uncreateGroup(linesOut2))
        self.play(Uncreate(out2), output.animate.shift(DOWN*1.3))
        lines = connectLayers(layer4, output)
        drawArr(self, lines, RED, 1)
        percent = Text(".35", color = WHITE, font_size = 30)
        perval = Text("35%", color = WHITE, font_size = 30)
        perval.move_to(output)
        perval.shift(RIGHT*3)
        percent.move_to(output)
        self.play(Create(percent))
        self.play(Transform(percent, perval))
        yval = MathTex(r"y", color=WHITE, font_size = 60)
        yval.move_to(output)
        self.play(Create(yval))
        self.play(Uncreate(percent))
        deLines2 = uncreateGroup(lines)
        out2 = Circle()
        out2.move_to(output)
        out2.set_color(WHITE)
        self.play(Create(out2), deLines2, out2.animate.shift(DOWN*1.3), output.animate.shift(UP*1.3), Uncreate(yval))
        lines3 = connectLayers(layer4, [out2, output])
        drawArr(self, lines3, PINK, 1)
        ys = [MathTex(r"y_{1}", color=WHITE, font_size = 60), MathTex(r"y_{2}", color=WHITE, font_size = 60)]
        ys[0].move_to(output)
        ys[1].move_to(out2)
        drawArr(self, ys, ORANGE, 1)
        self.wait(1)
        answ = [Text("Talk %", color = WHITE, font_size = 25), Text("Spam %", color = WHITE, font_size = 25)]
        answ[0].move_to(output)
        answ[1].move_to(out2)
        self.play(Transform(ys[0], answ[0]), Transform(ys[1], answ[1]))
        self.wait(1)
        outcomes = [Text("4.2%", color = WHITE, font_size = 25), Text("36.8%", color = WHITE, font_size = 25)]
        outcomes[0].move_to(output)
        outcomes[1].move_to(out2)
        outcomes[0].shift(RIGHT*3)
        outcomes[1].shift(RIGHT*3)
        self.play(Transform(ys[0], outcomes[0]), Transform(ys[1], outcomes[1]))
        self.wait(1)
        self.play(Uncreate(ys[0]), Uncreate(ys[1]), Uncreate(log))
        
        #Data Flow
        self.play(self.camera.frame.animate.move_to(layer3[2]).set(width = 25*circle.width))
        self.play(text.animate.set_color_by_tex('x_{1}', RED), xvals[0].animate.set_color_by_tex('x', RED))
        self.play(text.animate.set_color_by_tex('x_{1}', WHITE), xvals[0].animate.set_color_by_tex('x', WHITE))
        self.play(text.animate.set_color_by_tex('x_{2}', RED), xvals[1].animate.set_color_by_tex('x', RED))
        self.play(text.animate.set_color_by_tex('x_{2}', WHITE), xvals[1].animate.set_color_by_tex('x', WHITE))
        self.play(text.animate.set_color_by_tex('x_{3}', RED), xvals[2].animate.set_color_by_tex('x', RED))
        self.play(text.animate.set_color_by_tex('x_{3}', WHITE), xvals[2].animate.set_color_by_tex('x', WHITE))
        togreen = recolorGroup(self, xvals+ lines12, GREEN)
        towhite = recolorGroup(self, xvals+ lines12, WHITE)
        self.play(Uncreate(xvals[0]), Uncreate(xvals[1]), Uncreate(xvals[2]), togreen)
        newGroup = recolorGroup(self, layer2, GREEN)
        self.play(towhite, newGroup)
        hiddenBox = Rectangle(YELLOW, circle.height*5 + 2.5, circle.width*3+17)
        self.play(Create(hiddenBox))
        hiddenText = Text("Hidden Layers", color = YELLOW)
        hiddenText.shift(UP*circle.width*3.3)
        self.play(Create(hiddenText))
        
        nextLayer(self, layer2, layer3, lines23)
        nextLayer(self, layer3, layer4, lines34)
        nextLayer(self, layer4, [out2, output], lines3)
        
        results = [Text("0.46", color = WHITE, font_size = 30), Text("0.85", color = WHITE, font_size = 30)]
        results[0].move_to(output)
        results[1].move_to(out2)
        drawArr(self, results, WHITE, 1)
        self.play(output.animate.set_color(WHITE), results[0].animate.set_color(GREEN), results[1].animate.set_color(GREEN))
        self.wait(1)
        self.play(out2.animate.set_fill(GREEN, opacity = 0.5), results[0].animate.set_color(WHITE), results[1].animate.set_color(WHITE))
        self.wait(0.5)
        self.play(self.camera.frame.animate.move_to(layer4[2]).set(width = layer4[2].width*.5),  Unwrite(results[0]), Unwrite(results[1]), Uncreate(hiddenBox), Uncreate(hiddenText), out2.animate.set_color(WHITE))
        out2.set_fill(WHITE, opacity = 0.0),

        self.wait(112)

        #Scene Five
        self.play(self.camera.frame.animate.move_to(layer3[2]).set(width = layer3[2].width*25))
        nums = [MathTex(r"0.3", font_size = 100), MathTex(r"3.2", font_size = 100), MathTex(r"0", font_size = 100), MathTex(r"0.78", font_size = 100), MathTex(r"0", font_size = 100)]
        for num, node in zip(nums, layer3):
            num.move_to(node)
            self.play(Write(num))
        group = VGroup()
        group2 = VGroup()
        for num in nums:
            group = VGroup(group, num)
        for line in lines34:
            group2 = VGroup(group2, line)
        self.play(group.animate.set_color(GREEN))
        self.play(Unwrite(group), group2.animate.set_color(GREEN))
        self.play(self.camera.frame.animate.shift(RIGHT*45), group2.animate.set_color(WHITE))
        #Scene End

        self.wait(12)

        #Scene 7 Open
        self.play(self.camera.frame.animate.shift(LEFT*45))
        layerBox = Rectangle(YELLOW, circle.height*3 + 1.5, circle.width+1)
        layerBox.move_to(circles[1])
        inp =  Text("Input Layer", color = YELLOW)
        inp.move_to(circles[1])
        inp.shift(UP*circle.width*2.5)
        self.play(Write(inp), Create(layerBox))
        
        outputBox = Rectangle(YELLOW, circle.height*2 + 1.5, circle.width + 1)
        outputBox.move_to(output)
        out =  Text("Output Layer", color = YELLOW)
        outputBox.shift(DOWN*1.2)
        out.move_to(output)
        out.shift(UP*circle.width*1.5)
        self.play(Write(out), Create(outputBox))
        self.wait(1)
        
        self.play(layerBox.animate.set_color(GREEN), inp.animate.set_color(GREEN))
        self.play(xvals[0].animate.set_color(GREEN))
        self.play(xvals[1].animate.set_color(GREEN))
        self.play(xvals[2].animate.set_color(GREEN))
        x = VGroup(layerBox, inp, xvals[0], xvals[1], xvals[2])
        xl = VGroup()
        for line in lines12:
            xl = VGroup(xl, line)
        self.play(x.animate.set_color(WHITE), xl.animate.set_color(GREEN))
        self.play(xl.animate.set_color(WHITE))
        self.wait(1)
        xes = VGroup(xvals[0], xvals[1], xvals[2])
        cs = [MathTex(r"C\cdot", font_size = 120), MathTex(r"C\cdot", font_size = 120), MathTex(r"C\cdot", font_size = 120)]
        self.play(Unwrite(layerBox))
        cs[0].move_to(xvals[0])
        cs[1].move_to(xvals[1])
        cs[2].move_to(xvals[2])
        ces = VGroup(cs[0], cs[1], cs[2])
        ces.shift(LEFT*2+UP*0.2)
        self.play(Write(ces))
        self.wait(1)
        self.play(Unwrite(ces))
        self.play(Unwrite(inp))
        self.play(Unwrite(xes), self.camera.frame.animate.move_to(output).set(width = output.width*12))
        linearAct = Text("Linear Activation is cool here", font_size = 40)
        linearAct.move_to(out2)
        linearAct.shift(DOWN*3)
        self.play(Write(linearAct))
        ys = [MathTex(r"y_{1}", color=WHITE, font_size = 60), MathTex(r"y_{2}", color=WHITE, font_size = 60)]
        ys[0].move_to(output)
        ys[1].move_to(out2)
        drawArr(self, ys, ORANGE, 1)
        self.wait(1)
        yF = [MathTex(r"0.57", color=WHITE, font_size = 60).move_to(ys[0]), MathTex(r"0.83", color=WHITE, font_size = 60).move_to(ys[1])]
        self.play(Transform(ys[0], yF[0]), Transform(ys[1], yF[1]))
        self.wait(1)
        yFMult = [MathTex(r"0.57*C_{1}", color=WHITE, font_size = 30).move_to(ys[0]), MathTex(r"0.83*C_{2}", color=WHITE, font_size = 30).move_to(ys[1])]
        self.play(Transform(ys[0], yFMult[0]), Transform(ys[1], yFMult[1]))
        self.wait(1)
        yFPer = [MathTex(r"57\%", color=WHITE, font_size = 60).move_to(ys[0]), MathTex(r"83\%", color=WHITE, font_size = 60).move_to(ys[1])]
        self.play(Transform(ys[0], yFPer[0]), Transform(ys[1], yFPer[1]))
        self.wait(1)
        self.play(self.camera.frame.animate.move_to(layer3[2]).set(width = 25*circle.width))    
        

        uniGroup = VGroup(out, outputBox, output, ys[0], ys[1], linearAct, out2, text)
        for n in circles:
            uniGroup = VGroup(uniGroup, n)
        for n in layer2:
            uniGroup = VGroup(uniGroup, n)
        for n in layer3:
            uniGroup = VGroup(uniGroup, n)
        for n in layer4:
            uniGroup = VGroup(uniGroup, n)
        for n in lines12:
            uniGroup = VGroup(uniGroup, n)
        for n in lines23:
            uniGroup = VGroup(uniGroup, n)
        for n in lines34:
            uniGroup = VGroup(uniGroup, n)
        for n in lines45:
            uniGroup = VGroup(uniGroup, n)
        for n in lines3:
            uniGroup = VGroup(uniGroup, n)
        t = Text("I have no literally zero subscribers.", font_size = 120)
        t1 = Text("This is my first YouTube video", font_size = 120)
        t2 = Text("I don't even have a subscriber graph to show you", font_size = 120)
        self.play(Transform(uniGroup, t))
        self.wait(1)
        self.play(Transform(uniGroup, t1))
        self.wait(1)
        self.play(Transform(uniGroup, t2))
        self.wait(2)


def nextLayer(self, curLay, nextLay, lines):
    cw = recolorGroup(self, curLay, WHITE)
    lg = recolorGroup(self, lines, GREEN)
    gn = recolorGroup(self, nextLay, GREEN)
    lw = recolorGroup(self, lines, WHITE)
    self.play(cw, lg)
    self.play(gn, lw)
def recolorGroup(self, arr, color):
    group = AnimationGroup()
    for val in arr:
        group = AnimationGroup(group, val.animate.set_color(color))
    return group

def uncreateGroup(arr):
    group = AnimationGroup()
    for val in arr:
        group = AnimationGroup(Uncreate(val), group)
    return group

def wholeArrayColorSwitch(col, arr):
    group = AnimationGroup()
    for val in arr:
        group = AnimationGroup(group, val.animate.set_color(col))
    return group

def addXes(obj, arr):
    group = AnimationGroup()
    retArr = []
    for i, val in enumerate(arr):
        
        x = MathTex(r"x_{" + str(i + 1) + "}", font_size = 120)
        x.move_to(val)
        retArr.append(x)
        group = AnimationGroup(group, Create(x))
    obj.play(group)
    return retArr
        


def moveGroup(arr, distance):
    group = AnimationGroup()
    for val in arr:
        group = AnimationGroup(group, val.animate.shift(LEFT*distance))
    return group

def drawArr(obj, arr, drawColor, anispeed2):
    group = AnimationGroup()
    for val in arr:
        val.set_color(drawColor)
        group = AnimationGroup(group, Create(val))
        group = AnimationGroup(group, val.animate.set_color(WHITE))
    obj.play(group)

def makeLayer(obj, layer, numHiddens, offset):
    for x in range(0, numHiddens):
        layer.append(Circle())
        layer[x].set_color(WHITE)
        if(x == 0):
            layer[x].shift(DOWN*5+LEFT*offset)
        else:
            layer[x].next_to(prevCirc, UP, buff = 0.5)
        prevCirc = layer[x]

def connectLayers(layer1, layer2):
    lines = []
    for layer in layer1:
        for layere in layer2:
            lines.append(Line(layer, layere))
    return lines

