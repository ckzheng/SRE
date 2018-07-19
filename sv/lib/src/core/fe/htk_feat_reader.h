#ifndef HTK_FEAT_READER
#define HKT_FEAT_READER
#include <vector>
#include <algorithm>

namespace idec {

class HtkFeatReader {
 public:
  HtkFeatReader() {
  }

  static bool ReadHtkFeature(const char *feat_file, std::vector<float> &data_buf,
                             int &frm_num, int &dim) {
    std::exception e;
    FILE *fp = fopen(feat_file, "rb");
    if (!fp) {
      IDEC_ERROR << "cannot open feature file";
      return false;
    }
    // read meta-data
    int sample_period, frm_num_in;
    short sample_kind, nDim;
    RawReadInt(fp, &frm_num_in, 1, true, true);
    RawReadInt(fp, &sample_period, 1, true, true);
    RawReadShort(fp, &nDim, 1, true, true);
    RawReadShort(fp, &sample_kind, 1, true, true);
    nDim /= sizeof(float);

    frm_num = frm_num_in;
    dim = nDim;
    // read data
    data_buf.resize(nDim*frm_num_in);

    if (!RawReadFloat(fp, &(data_buf[0]), nDim*frm_num_in, true, true)) {
      IDEC_ERROR << "error allocating memory when loading feature";
      throw e;
    }

    fclose(fp);
    return true;
  }

 private:
  ////////////////////////////////////////////////////////
  //  set of local functions for parse htk-format feature
  /////////////////////////////////////////////////////////

  /* SwapInt32: swap byte order of int32 data value *p */
  static void SwapInt32(int *p) {
    char temp, *q;

    q = (char *)p;
    temp = *q;
    *q = *(q + 3);
    *(q + 3) = temp;
    temp = *(q + 1);
    *(q + 1) = *(q + 2);
    *(q + 2) = temp;
  }

  /* SwapShort: swap byte order of short data value *p */
  static void SwapShort(short *p) {
    char temp, *q;

    q = (char *)p;
    temp = *q;
    *q = *(q + 1);
    *(q + 1) = temp;
  }


  /* EXPORT->ReadShort: read n short's from src in ascii or binary */
  static bool
  RawReadShort(FILE *src, short *s, int n, bool bin, bool swap) {
    int j, k, count = 0, x;
    short *p;

    if (bin) {
      if (fread(s, sizeof(short), n, src) != n)
        return false;
      if (swap)
        for (p = s, j = 0; j < n; p++, j++)
          SwapShort(p);  /* Need to swap to machine order */

      count = n*sizeof(short);
    } else {
      for (j = 1; j <= n; j++) {
        if (fscanf(src, "%d%n", &x, &k) != 1)
          return false;
        *s++ = x;
        count += k;
      }
    }

    return true;
  }

  /* EXPORT->ReadInt: read n ints from src in ascii or binary */
  static bool
  RawReadInt(FILE *src, int *i, int n, bool bin, bool swap) {
    int j, k, count = 0;
    int *p;

    if (bin) {
      if (fread(i, sizeof(int), n, src) != n)
        return false;
      if (swap)
        for (p = i, j = 0; j < n; p++, j++)
          SwapInt32((int *)p); /* Read in SUNSO unless natReadOrder=T */

      count = n*sizeof(int);
    } else {
      for (j = 1; j <= n; j++) {
        if (fscanf(src, "%d%n", i, &k) != 1)
          return false;
        i++;
        count += k;
      }
    }
    return true;
  }

  static bool
  RawReadFloat(FILE *src, float *x, int n, bool bin, bool swap) {
    int k, count = 0, j;
    float *p;

    if (bin) {
      if (fread(x, sizeof(float), n, src) != n)
        return false;
      if (swap) {
        for (p = x, j = 0; j < n; p++, j++)
          SwapInt32((int *)p); /* Read in SUNSO unless natReadOrder=T */
      }

      count += n*sizeof(float);
    } else {
      for (j = 1; j <= n; j++) {
        if (fscanf(src, "%e%n", x, &k) != 1)
          return false;
        x++;
        count += k;
      }
    }
    return true;
  }
};
}
#endif

