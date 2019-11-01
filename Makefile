GCC = g++
CFLAGS =-I include/
LDFLAGS = -lGL -lGLEW -lglfw -lOpenCL

PROGRAM = test

ODIR = obj
BIN = bin
IDIR = src
SDIR = as

OUT = $(BIN)/$(PROGRAM)

all: $(IDIR) $(ODIR) $(BIN) $(OUT)

run: all
	./$(BIN)/$(PROGRAM)
$(ODIR):
	mkdir $(ODIR)
$(BIN):
	mkdir $(BIN)
$(IDIR):
	mkdir $(IDIR)
$(SDIR):
	mkdir $(SDIR)

C_SOURCES = $(shell find $(IDIR) -type f -name *.cpp -printf "%f\n")
OBJECTS = $(patsubst %.cpp, $(ODIR)/%.o,$(C_SOURCES))
S_CODE = $(patsubst %.cpp, $(SDIR)/%.s,$(C_SOURCES))

$(ODIR)/%.o : $(IDIR)/%.cpp
	$(GCC) $(CFLAGS) -c $^ -o $@

$(SDIR)/%.s : $(IDIR)/%.cpp
	$(GCC) -g -o $@ $(CFLAGS) -S $^

$(ODIR)/%.o : $(IDIR)/**/%.cpp
	$(GCC) $(CFLAGS) -c $^ -o $@

$(SDIR)/%.s : $(IDIR)/**/%.cpp
	$(GCC) -g -o $@ $(CFLAGS) -S $^


$(OUT): $(OBJECTS)
	$(GCC) $(LDFLAGS) -o $(BIN)/$(PROGRAM) $(OBJECTS)

clean: $(ODIR) $(BIN)
	rm -rf $(ODIR)
	rm -rf $(BIN)
	rm -rf $(SDIR)

clean-dis: $(SDIR)
	rm -rf $(SDIR)

dis: all $(SDIR) $(S_CODE)
