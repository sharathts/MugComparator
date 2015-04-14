# MugComparator
Given a test image, obtains 10 similar images from the database. Also has reinforcement learning capability if the user points out that one of the faces is not in the group.
# Required Libraries
1. Numpy
2. OpenCV
3. Django

# Instructions to run
1. Download the zip file and extract it.
2. Extract the data.tar.gz in the same folder as the code.
3. Type python main.py <location of test image> to run the code
4. A new file called results.html will be created through which results can be viewed.
5. The program is still running and is waiting for user based reinforcement. Press Ctrl+C to quit training the system. Or enter space seperated numbers corresponding to the images which are not similar and the system learns from this.

