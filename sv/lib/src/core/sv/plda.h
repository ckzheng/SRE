#ifndef _PLDA_H_
#define _PLDA_H_

#include "new_vector.h"
#include "new_matrix.h"
#include "base/log_message.h"
#include "resource_manager.h"

class Plda {
 public:
  Plda(const PldaOptions &config,
       const PldaResource &plda_res) :plda_opt_(config), mean_(plda_res.mean),
    transform_(plda_res.transform), psi_(plda_res.psi),
    offset_(plda_res.offset) {}

  int Dim() const {return mean_.Size();}

  double GetNormalizationFactor(const DoubleVector &transformed_ivector,
                                int num_examples) const;
  double TransformIvector(const DoubleVector &ivector, int num_examples,
                          DoubleVector &transformed_ivector) const;
  double LogLikelihoodRatio(const DoubleVector &train_ivector,
                            int n, const DoubleVector &test_ivector) const;
 private:
  const DoubleVector &mean_;
  const DoubleMatrix &transform_;
  const DoubleVector &psi_;
  const DoubleVector &offset_;
  const PldaOptions &plda_opt_;
};

#endif // !_PLDA_H_

