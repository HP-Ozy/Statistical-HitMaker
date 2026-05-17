CC      = gcc
CFLAGS  = -O2 -Wall -fPIC
LDFLAGS = -shared
LIBS    = -lsndfile
TARGET  = libaudio.so
SRC     = audio_preprocessor.c

all: $(TARGET)
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) $(SRC) $(LIBS)
	@echo "Compilato: $(TARGET)"
clean:
	rm -f $(TARGET)
