#include "indeseminb_new.h"
#include "utils.h"
#include "correlationMeasures.h"
#include <assert.h>
#include <math.h>
#include <set>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
using namespace std;

indeseminb_new::indeseminb_new() :
trainingIsFinished_(false) {
}

indeseminb_new::indeseminb_new(char* const *&, char* const *) :
xxyDist_(), trainingIsFinished_(false) {
    name_ = "indeseminb_new";
}

indeseminb_new::~indeseminb_new(void) {
}

void indeseminb_new::reset(InstanceStream &is) {
    instanceStream_ = &is;
    const unsigned int noCatAtts = is.getNoCatAtts();
    noCatAtts_ = noCatAtts;
    noClasses_ = is.getNoClasses();
    trainingIsFinished_ = false;
    parents_.resize(noCatAtts);
    Hmin.resize(noCatAtts);
    parents_used.resize(noCatAtts);
    Hmin_used.resize(noCatAtts);
    classDist_new.resize(noClasses_);
    for (CategoricalAttribute a = 0; a < noCatAtts_; a++) {
        parents_[a] = NOPARENT;
        parents_used[a] = NOPARENT;
    }
    xxyDist_.reset(is);
    xxyDist_new.reset(is);
}

void indeseminb_new::getCapabilities(capabilities &c) {
    c.setCatAtts(true); // only categorical attributes are supported at the moment
}

void indeseminb_new::initialisePass() {
    assert(trainingIsFinished_ == false);
}

void indeseminb_new::train(const instance &inst) {
    xxyDist_.update(inst);
    inst_.push_back(inst);
}

void indeseminb_new::classify(const instance &inst, std::vector<double> &classDist) {

    for (CatValue y = 0; y < noClasses_; y++) {
        classDist[y] = xxyDist_.xyCounts.p(y);
    }
    for (CatValue y = 0; y < noClasses_; y++) {
        for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
            const CategoricalAttribute parent = parents_used[x1];
            double pxy = xxyDist_.xyCounts.p(x1, inst.getCatVal(x1), y);
            if (parent == NOPARENT) {
                classDist[y] *= pxy;

            } else {
                double pxxy = xxyDist_.p(x1, inst.getCatVal(x1), parent,
                        inst.getCatVal(parent), y);
                classDist[y] *= pxxy;

            }
        }
    }
    normalise(classDist);
}

void indeseminb_new::finalisePass() {
    assert(trainingIsFinished_ == false);
    for (int i = 0; i < inst_.size() / 2; i++) {

    }

    for (float threshold = 0.0; threshold < 1.1; threshold = threshold + 0.1) {
        float errorcount = 0.0;
        for (unsigned int i = 0; i < inst_.size(); i++) {
            xxyDist_new.clear();
            for (unsigned int l = 0; l < inst_.size(); l++) {
                xxyDist_new.update(inst_[i]);
            }
            crosstab<float> hi = crosstab<float>(noCatAtts_);
            crosstab<float> cmi = crosstab<float>(noCatAtts_);
            getCondMutualInf(xxyDist_, cmi);
            getEntropyInf(xxyDist_, hi);

            for (unsigned int j = 0; j < noCatAtts_; j++) {
                parents_[j] = NOPARENT;
            }
            std::set<CategoricalAttribute> innet;
            std::set<CategoricalAttribute> free;
            innet.clear();
            free.clear();
            unsigned int x1 = 0;
            unsigned int x2 = 0;
            float minhi = std::numeric_limits<double>::max();
            float maxmi = 0.0;
            for (unsigned int i = 0; i < noCatAtts_; i++) {
                for (unsigned int j = 0; j < noCatAtts_; j++) {
                    if (i == j)
                        continue;
                    else {
                        Hmin[j] += hi[i][j];
                    }
                }
            }
            for (unsigned int i = 0; i < noCatAtts_; i++) {
                if (minhi > Hmin[i]) {
                    minhi = Hmin[i];
                    x1 = i;
                }
            }
            float maxcmi = 0.0;
            for (unsigned int i = 1; i < noCatAtts_; i++) {
                for (unsigned int j = 0; j < noCatAtts_; j++) {
                    if (cmi[i][j] > maxcmi) {
                        maxcmi = cmi[i][j];
                    }
                }
            }
            innet.insert(x1);
            for (unsigned int i = 0; i < noCatAtts_; i++) {
                if (i == x1)
                    continue;
                else
                    free.insert(i);
            }
            float threcmi = threshold * maxcmi;
            while (!free.empty()) {
                float a = std::numeric_limits<float>::max();
                Hmin.assign(noCatAtts_, 0.0);
                for (std::set<CategoricalAttribute>::const_iterator j = free.begin(); j != free.end(); j++) {
                    Hmin[*j] = 0.0;
                }
                for (std::set<CategoricalAttribute>::const_iterator i = free.begin(); i != free.end(); i++) {
                    for (std::set<CategoricalAttribute>::const_iterator j = free.begin(); j != free.end(); j++) {
                        if (i == j)
                            continue;
                        else {
                            Hmin[*i] += hi[*j][*i];
                        }
                    }
                }
                unsigned int xk = NOPARENT;
                float Hminmin = std::numeric_limits<float>::max();
                for (std::set<CategoricalAttribute>::const_iterator j = free.begin(); j != free.end(); j++) {
                    if (Hminmin > Hmin[*j]) {
                        Hminmin = Hmin[*j];
                        xk = *j;
                    }
                }
                float cmimax = -std::numeric_limits<float>::max();
                unsigned int xp = NOPARENT;
                bool dependence = false;
                for (std::set<CategoricalAttribute>::const_iterator i = innet.begin(); i != innet.end(); i++) {
                    if (cmimax < cmi[xk][*i]) {
                        cmimax = cmi[xk][*i];
                        xp = *i;
                    }
                }
                if (cmimax > threcmi) {
                    parents_[xk] = xp;
                }
                innet.insert(xk);
                free.erase(xk);
            }
            unsigned int trueclass = inst_[i].getClass();
            for (CatValue y = 0; y < noClasses_; y++) {
                classDist_new[y] = xxyDist_new.xyCounts.p(y);
            }
            for (CatValue y = 0; y < noClasses_; y++) {
                for (unsigned int x1 = 0; x1 < noCatAtts_; x1++) {
                    const CategoricalAttribute parent = parents_[x1];
                    double pxy = xxyDist_new.xyCounts.p(x1, inst_[i].getCatVal(x1), y);
                    if (parent == NOPARENT) {
                        classDist_new[y] *= pxy;

                    } else {
                        double pxxy = xxyDist_new.p(x1, inst_[i].getCatVal(x1), parent,
                                inst_[i].getCatVal(parent), y);
                        classDist_new[y] *= pxxy;

                    }
                }
            }
            normalise(classDist_new);
            unsigned int prediction = 0;
            double maxclass = 0.0;
            for (CatValue y = 0; y < noClasses_; y++) {
                if (classDist_new[y] > maxclass) {
                    maxclass = classDist_new[y];
                    prediction = y;
                }
            }
            if (prediction != trueclass)
                errorcount = errorcount + 1;

            int lo = static_cast<int> (threshold * 10);
            eachclassDist[lo] = errorcount / static_cast<float> (inst_.size());
        }
    }
    float minloss = 1.0;
    int threshold_used = 0;
    for (int i = 0; i < 11; i++) {
        if (eachclassDist[i] < minloss) {
            minloss = eachclassDist[i];
            threshold_used = i;
        }
    }
    for (int i = 0; i < 11; i++) {
        cout << eachclassDist[i] << '\t';
    }
    cout << endl;
    cout << threshold_used << endl;
    std::set<CategoricalAttribute> innet_used;
    std::set<CategoricalAttribute> free_used;
    innet_used.clear();
    free_used.clear();
    unsigned int x1 = 0;
    unsigned int x2 = 0;
    float minhi_used = std::numeric_limits<double>::max();
    float maxmi_used = 0.0;
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        for (unsigned int j = 0; j < noCatAtts_; j++) {
            if (i == j)
                continue;
            else {
                Hmin_used[j] += hi[i][j];
            }
        }
    }
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        if (minhi_used > Hmin_used[i]) {
            minhi_used = Hmin_used[i];
            x1 = i;
        }
    }
    float maxcmi_used = 0.0;
    for (unsigned int i = 1; i < noCatAtts_; i++) {
        for (unsigned int j = 0; j < noCatAtts_; j++) {
            if (cmi[i][j] > maxcmi_used) {
                maxcmi_used = cmi[i][j];
            }
        }
    }
    innet_used.insert(x1);
    for (unsigned int i = 0; i < noCatAtts_; i++) {
        if (i == x1)
            continue;
        else
            free_used.insert(i);
    }
    float threcmi = (static_cast<float> (threshold_used) / 10) * maxcmi_used;
    while (!free_used.empty()) {
        float a = std::numeric_limits<float>::max();
        Hmin_used.assign(noCatAtts_, 0.0);
        //cout<<2<<endl;
        for (std::set<CategoricalAttribute>::const_iterator j = free_used.begin(); j != free_used.end(); j++) {
            Hmin_used[*j] = 0.0;
        }
        for (std::set<CategoricalAttribute>::const_iterator i = free_used.begin(); i != free_used.end(); i++) {
            for (std::set<CategoricalAttribute>::const_iterator j = free_used.begin(); j != free_used.end(); j++) {
                if (i == j)
                    continue;
                else {
                    Hmin_used[*i] += hi[*j][*i];
                }
            }
        }
        unsigned int xk = NOPARENT;
        float Hminmin_used = std::numeric_limits<float>::max();
        for (std::set<CategoricalAttribute>::const_iterator j = free_used.begin(); j != free_used.end(); j++) {
            if (Hminmin_used > Hmin_used[*j]) {
                Hminmin_used = Hmin_used[*j];
                xk = *j;
            }
        }
        float cmimax = -std::numeric_limits<float>::max();
        unsigned int xp = NOPARENT;
        bool dependence = false;
        for (std::set<CategoricalAttribute>::const_iterator i = innet_used.begin(); i != innet_used.end(); i++) {
            if (cmimax < cmi[xk][*i]) {
                cmimax = cmi[xk][*i];
                xp = *i;
            }
        }
        if (cmimax > threcmi) {
            parents_used[xk] = xp;
        }
        innet_used.insert(xk);
        free_used.erase(xk);
    }
    trainingIsFinished_ = true;
}

/// true iff no more passes are required. updated by finalisePass()

bool indeseminb_new::trainingIsFinished() {
    return trainingIsFinished_;
}

