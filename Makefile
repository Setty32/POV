# Nazvy zdrojovych souboru pro jednotlive casti ukolu
SRC=main.cpp

# Nazvy vystupnich souboru pro jednotlive casti ukolu
TARGET=stitcher

TARGET_FILES=shanghai.jpg mountains.jpg
MOUNTAIN_FILES = 1_1.jpg 2_1.jpg 3_1.jpg 4_1.jpg
SHANGHAI_FILES = shanghai01.jpg shanghai02.jpg shanghai03.jpg

DOC_FILES = doc/img doc/Makefile doc/Makefile.mk doc/czplain.bst doc/dokumentace.pdf \
    doc/dokumentace.tex doc/img/logo-eps-converted-to.pdf doc/img/logo.eps doc/prezentace.tex \
    doc/title.tex doc/zdroje.bib


# Parametry prekladace
CC=g++ 
CFLAGS=`pkg-config --cflags opencv` -std=gnu++0x -g
LIBS=`pkg-config --libs opencv`

.PHONY: $(TARGET_FILES) pack

# Kompilace obou detektoru
all: $(TARGET)

$(TARGET): $(SRC) 
	$(CC) $(CFLAGS)  $^ -o $@ $(LIBS)

clean:
	rm -f *.o *~ $(TARGET) $(TARGET_FILES)

shanghai.jpg: $(TARGET)
	./$(TARGET) $(SHANGHAI_FILES) $@

mountains.jpg: $(TARGET)
	./$(TARGET) $(MOUNTAIN_FILES) $@
	
pack:
	tar cvzf project.tar.gz $(SRC) $(SHANGHAI_FILES) $(MOUNTAIN_FILES) Makefile README.md $(DOC_FILES)

