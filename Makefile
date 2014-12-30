# Nazvy zdrojovych souboru pro jednotlive casti ukolu
SRC_FILE2=main.cpp

# Nazvy vystupnich souboru pro jednotlive casti ukolu
TARGET2=main

# Nastavte vstupnĂ­ parametry potĹebnĂŠ pro spuĹĄtÄnĂ­ aplikace, pokud chcete spouĹĄtÄt pomoci Makefile
TARGET2_ARGUMENTS= IMG2.jpg IMG1.jpg chimney.jpg
TARGET2_ARGUMENTS1= 1.JPG 2.JPG 3.JPG catedral.jpg #4.JPG 5.JPG 6.JPG
TARGET2_ARGUMENTS2= 1_1.jpg 2_1.jpg 3_1.jpg 4_1.jpg mountains.jpg  #5_1.jpg #6_1.jpg 7_1.jpg
TARGET2_SHNAG = shanghai01.jpg shanghai02.jpg shanghai03.jpg shanghai.jpg
# Parametry prekladace
CC=g++ 
CFLAGS=`pkg-config --cflags opencv` -std=gnu++0x
LIBS=`pkg-config --libs opencv`

# Kompilace obou detektoru
all: $(TARGET2)

$(TARGET2): $(SRC_FILE2) 
	$(CC) $(CFLAGS)  $^ -o $@ $(LIBS)


clean:
	rm -f *.o *~ $(TARGET2)

run1: $(TARGET2)
	./$(TARGET2) $(TARGET2_ARGUMENTS1)	

run2: $(TARGET2)
	./$(TARGET2) $(TARGET2_ARGUMENTS)

run3: $(TARGET2)
	./$(TARGET2) $(TARGET2_ARGUMENTS2)

shang: $(TARGET2)
	./$(TARGET2) $(TARGET2_SHNAG)
