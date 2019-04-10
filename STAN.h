/* 
 * File:   indeseminb_new.h
 * Author: dell
 *
 * Created on 2018年5月27日, 下午7:56
 */
#pragma once

#include "incrementalLearner.h"
#include "xxyDist.h"
//#include "xxxyDist.h"
#include <limits>

class indeseminb_new: public IncrementalLearner {
public:
	indeseminb_new();
	indeseminb_new(char* const *& argv, char* const * end);
	~indeseminb_new(void);

	void reset(InstanceStream &is);   ///< reset the learner prior to training
	void initialisePass(); ///< must be called to initialise a pass through an instance stream before calling train(const instance). should not be used with train(InstanceStream)
	void train(const instance &inst); ///< primary training method. train from a single instance. used in conjunction with initialisePass and finalisePass
	void finalisePass(); ///< must be called to finalise a pass through an instance stream using train(const instance). should not be used with train(InstanceStream)
	bool trainingIsFinished(); ///< true iff no more passes are required. updated by finalisePass()
	void getCapabilities(capabilities &c);

	virtual void classify(const instance &inst, std::vector<double> &classDist);

private:
	unsigned int noCatAtts_;          ///< the number of categorical attributes.
	unsigned int noClasses_;                          ///< the number of classes

	InstanceStream* instanceStream_;
	std::vector<CategoricalAttribute> parents_;
        std::vector<float> Hmin;
        std::vector<CategoricalAttribute> parents_used;
        std::vector<float> Hmin_used;
        std::vector<instance> inst_;
        std::vector<double> classDist_new;
        float eachclassDist[11];
	xxyDist xxyDist_;
        xxyDist xxyDist_new;
	bool trainingIsFinished_; ///< true iff the learner is trained
	const static CategoricalAttribute NOPARENT = 0xFFFFFFFFUL; //使用printf("%d",0xFFFFFFFFUL);输出是-1 cannot use std::numeric_limits<categoricalAttribute>::max() because some compilers will not allow it here
};

