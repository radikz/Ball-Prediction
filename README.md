# Ball Prediction Using Kalman Filter

## Requirements
* Library OpenCV
* Library Eigen

---
## How to run and compile
```cli
g++ main.cpp -o main `pkg-config --cflags --libs opencv` && ./main
```

## Setup
```cpp
cv::VideoCapture cap("your source video.mp4");
```
---

**Ball prediction**
![result](/Screenshots/result.jpg)
