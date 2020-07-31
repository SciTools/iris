SUBDIRS = src

html:
	@for i in $(SUBDIRS); do \
        echo "make html in $$i..."; \
        (cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) html); done

all:
	@for i in $(SUBDIRS); do \
		echo "make all in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) all); done

install:
	@for i in $(SUBDIRS); do \
		echo "Installing in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) install); done

build:
	@for i in $(SUBDIRS); do \
		echo "Clearing in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) build); done

clean:
	@for i in $(SUBDIRS); do \
		echo "Clearing in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) clean); done

doctest:
	@for i in $(SUBDIRS); do \
        echo "Running doctest in $$i..."; \
        (cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) doctest); done

linkcheck:
	@for i in $(SUBDIRS); do \
		echo "Running linkcheck in $$i..."; \
		(cd $$i; $(MAKE) $(MFLAGS) $(MYMAKEFLAGS) linkcheck); done

gallerytest:
	@echo
	@echo "Running \"gallery\" tests..."
	@echo
	python -m unittest discover -v -t . 
