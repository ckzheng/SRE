AM_CPPFLAGS = -I$(srcdir)/.. -I../../../../../include -I../../../../../asr/decoder/include -I../../../../../asr/decoder/src/core/ -msse3 -DUSE_SSE -DPOSIX -DHAVE_POSIX_MEMALIGN

noinst_LTLIBRARIES = libals_spkdiar.la
libals_spkdiar_la_SOURCES = \
	abstract_seg.cpp \
	ahc.cpp \
	bic.cpp \
	full_gaussian.cpp \
	mfcc.cpp \
	seg_cluster.cpp \
	seg.cpp \
	speaker_cluster.cpp \
	spkdiar_impl.cpp \
	spkdiar_tunning.cpp \
	viterbi.cpp \
	wav.cpp \
        spkdiar_serialize.cpp \
        vad.cpp \
        spkdiar.cpp

libals_spkdiar_la_LDFLAGS = -version-info 1:0:0


libals_spkdiar_la_LIBADD = \
        $(top_srcdir)/asr/decoder/src/libalsr/libalsr.la
       


