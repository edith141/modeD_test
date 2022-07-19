- install cmake
	- sudo dnf install cmake
	- cmake --version
- install dlib
	- http://dlib.net/files/dlib-19.24.tar.bz2
	- wget http://dlib.net/files/dlib-19.24.tar.bz2
	- tar xvf dlib-19.24.tar.bz2
	- cd dlib-19.24
	- mkdir build
	- cd build
	- cmake ..
	- sudo cmake --build . --target  install
	- sudo make install
	- sudo ldconfig

- compile with g++ cppfile.cpp dliba_library_file.a -o output_file_name.out

- alt compile command
	- g++ -std=c++11 -O3 -lpthread -lX11 view_net.cpp libdlib.a
	- g++ -std=c++11 -O3 -lpthread -lX11 file_name.cpp -I/usr/local/include -L/usr/local/lib64 -ldlib /usr/lib64/libX11.so 
	- g++ -std=c++11 -O3 -I../home/dlib-19.9/dlib/all/source.cpp -lpthread -lX11 file_name.cpp -I/usr/local/include -L/usr/local/lib64 -ldlib /usr/lib64/libX11.so 

- models
	- http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
	- http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

- extract models
	- bzip2 -dk filename.bz2

- extract tar bz files
	- tar -xvzf file_name.tar.gz

=================================================================================

