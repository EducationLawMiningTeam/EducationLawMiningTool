// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_OPTIMIZER_TINY_HPP
#define OPERON_OPTIMIZER_TINY_HPP

#include <Eigen/Core>
#include "operon/interpreter/interpreter.hpp"

namespace Operon {

// this cost function is adapted to work with both solvers from Ceres: the normal one and the tiny solver
// for this, a number of template parameters are necessary:
// - the AutodiffCalculator will compute and return the Jacobian matrix
// - the StorageOrder specifies the format of the jacobian (row-major for the big Ceres solver, column-major for the tiny solver)

template<typename DTable, int StorageOrder = Eigen::ColMajor>
struct CostFunction {
    static auto constexpr Storage{ StorageOrder };
    using Scalar = Operon::Scalar;

    enum {
        NUM_RESIDUALS = Eigen::Dynamic,  // NOLINT
        NUM_PARAMETERS = Eigen::Dynamic, // NOLINT
    };

    explicit CostFunction(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Span<Operon::Scalar const> target, Operon::Range const range, DTable const& dtable)
        : tree_{tree}
        , dataset_{dataset}
        , target_{target}
        , range_{range}
        , dtable_{dtable}
        , numResiduals_{range.Size()}
        , numParameters_{ParameterCount(tree)}
    { }

    inline auto Evaluate(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool // NOLINT
    {
        EXPECT(target_.size() == numResiduals_);
        EXPECT(parameters != nullptr);
        Operon::Span<Operon::Scalar const> params{ parameters, numParameters_ };

        Interpreter<Operon::Scalar, DTable> interpreter{dtable_.get(), dataset_, tree_};

        if (jacobian != nullptr) {
            Operon::Span<Operon::Scalar> jac{jacobian, static_cast<size_t>(numResiduals_ * numParameters_)};
            interpreter.JacRev(params, range_, jac);
        }

        if (residuals != nullptr) {
            Operon::Span<Operon::Scalar> res{ residuals, static_cast<size_t>(numResiduals_) };
            interpreter.Evaluate(params, range_, res);
            Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> x(residuals, numResiduals_);
            Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> y(target_.data(), numResiduals_);
            x -= y;
        }
        return true;
    }

    // ceres solver - jacobian must be in row-major format
    // tiny solver - jacobian must be in col-major format
    auto operator()(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const -> bool
    {
        return Evaluate(parameters, residuals, jacobian);
    }

    [[nodiscard]] auto NumResiduals() const -> int { return numResiduals_; }
    [[nodiscard]] auto NumParameters() const -> int { return numParameters_; }

    // required by Eigen::LevenbergMarquardt
    using JacobianType = Eigen::Matrix<Operon::Scalar, -1, -1>;
    using QRSolver     = Eigen::ColPivHouseholderQR<JacobianType>;

    // there is no real documentation but looking at Eigen unit tests, these functions should return zero
    // see: https://gitlab.com/libeigen/eigen/-/blob/master/unsupported/test/NonLinearOptimization.cpp
    auto operator()(Eigen::Matrix<Scalar, -1, 1> const& input, Eigen::Matrix<Scalar, -1, 1>& residual) const -> int
    {
        Evaluate(input.data(), residual.data(), nullptr);
        return 0;
    }

    auto df(Eigen::Matrix<Scalar, -1, 1> const& input, Eigen::Matrix<Scalar, -1, -1>& jacobian) const -> int // NOLINT
    {
        static_assert(StorageOrder == Eigen::ColMajor, "Eigen::LevenbergMarquardt requires the Jacobian to be stored in column-major format.");
        Evaluate(input.data(), nullptr, jacobian.data());
        //std::cout << "jacobian:\n" << jacobian << "\n";
        return 0;
    }

    [[nodiscard]] auto values() const -> int { return NumResiduals(); }  // NOLINT
    [[nodiscard]] auto inputs() const -> int { return NumParameters(); } // NOLINT

private:
    std::reference_wrapper<Tree const> tree_;
    std::reference_wrapper<Dataset const> dataset_;
    std::reference_wrapper<DTable const> dtable_;

    Operon::Range const range_; // NOLINT
    Operon::Span<Operon::Scalar const> target_;
    std::size_t numResiduals_;
    std::size_t numParameters_;

    inline auto ParameterCount(auto const& tree) const -> std::size_t {
        auto const& nodes = tree.Nodes();
        return std::count_if(nodes.cbegin(), nodes.cend(), [](auto const& n) { return n.Optimize; });
    }
};
} // namespace Operon

#endif
