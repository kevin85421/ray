.. _tune-main:

Ray Tune: Hyperparameter Tuning
===============================

.. toctree::
    :hidden:

    Getting Started <getting-started>
    Key Concepts <key-concepts>
    tutorials/overview
    examples/index
    faq
    api/api

.. image:: images/tune_overview.png
    :scale: 50%
    :align: center

Tune is a Python library for experiment execution and hyperparameter tuning at any scale.
You can tune your favorite machine learning framework (:ref:`PyTorch <tune-pytorch-cifar-ref>`, :ref:`XGBoost <tune-xgboost-ref>`, :doc:`TensorFlow and Keras <examples/tune_mnist_keras>`, and :doc:`more <examples/index>`) by running state of the art algorithms such as :ref:`Population Based Training (PBT) <tune-scheduler-pbt>` and :ref:`HyperBand/ASHA <tune-scheduler-hyperband>`.
Tune further integrates with a wide range of additional hyperparameter optimization tools, including :doc:`Ax <examples/ax_example>`, :doc:`BayesOpt <examples/bayesopt_example>`, :doc:`BOHB <examples/bohb_example>`, :doc:`Nevergrad <examples/nevergrad_example>`, and :doc:`Optuna <examples/optuna_example>`.

**Click on the following tabs to see code examples for various machine learning frameworks**:

.. tab-set::

    .. tab-item:: Quickstart

        To run this example, install the following: ``pip install "ray[tune]"``.

        In this quick-start example you `minimize` a simple function of the form ``f(x) = a**2 + b``, our `objective` function.
        The closer ``a`` is to zero and the smaller ``b`` is, the smaller the total value of ``f(x)``.
        We will define a so-called `search space` for  ``a`` and ``b`` and let Ray Tune explore the space for good values.

        .. callout::

            .. literalinclude:: ../../../python/ray/tune/tests/example.py
               :language: python
               :start-after: __quick_start_begin__
               :end-before: __quick_start_end__

            .. annotations::
                <1> Define an objective function.

                <2> Define a search space.

                <3> Start a Tune run and print the best result.


    .. tab-item:: Keras+Hyperopt

        To tune your Keras models with Hyperopt, you wrap your model in an objective function whose ``config`` you
        can access for selecting hyperparameters.
        In the example below we only tune the ``activation`` parameter of the first layer of the model, but you can
        tune any parameter of the model you want.
        After defining the search space, you can simply initialize the ``HyperOptSearch`` object and pass it to ``run``.
        It's important to tell Ray Tune which metric you want to optimize and whether you want to maximize or minimize it.

        .. callout::

            .. literalinclude:: doc_code/keras_hyperopt.py
                :language: python
                :start-after: __keras_hyperopt_start__
                :end-before: __keras_hyperopt_end__

            .. annotations::
                <1> Wrap a Keras model in an objective function.

                <2> Define a search space and initialize the search algorithm.

                <3> Start a Tune run that maximizes accuracy.

    .. tab-item:: PyTorch+Optuna

        To tune your PyTorch models with Optuna, you wrap your model in an objective function whose ``config`` you
        can access for selecting hyperparameters.
        In the example below we only tune the ``momentum`` and learning rate (``lr``) parameters of the model's optimizer,
        but you can tune any other model parameter you want.
        After defining the search space, you can simply initialize the ``OptunaSearch`` object and pass it to ``run``.
        It's important to tell Ray Tune which metric you want to optimize and whether you want to maximize or minimize it.
        We stop tuning this training run after ``5`` iterations, but you can easily define other stopping rules as well.


        .. callout::

            .. literalinclude:: doc_code/pytorch_optuna.py
                :language: python
                :start-after: __pytorch_optuna_start__
                :end-before: __pytorch_optuna_end__

            .. annotations::
                <1> Wrap a PyTorch model in an objective function.

                <2> Define a search space and initialize the search algorithm.

                <3> Start a Tune run that maximizes mean accuracy and stops after 5 iterations.

With Tune you can also launch a multi-node :ref:`distributed hyperparameter sweep <tune-distributed-ref>`
in less than 10 lines of code.
And you can move your models from training to serving on the same infrastructure with `Ray Serve`_.

.. _`Ray Serve`: ../serve/index.html


.. grid:: 1 2 3 4
    :gutter: 1
    :class-container: container pb-3

    .. grid-item-card::

        **Getting Started**
        ^^^

        In our getting started tutorial you will learn how to tune a PyTorch model
        effectively with Tune.

        +++
        .. button-ref:: tune-tutorial
            :color: primary
            :outline:
            :expand:

            Get Started with Tune

    .. grid-item-card::

        **Key Concepts**
        ^^^

        Understand the key concepts behind Ray Tune.
        Learn about tune runs, search algorithms, schedulers and other features.

        +++
        .. button-ref:: tune-60-seconds
            :color: primary
            :outline:
            :expand:

            Tune's Key Concepts

    .. grid-item-card::

        **User Guides**
        ^^^

        Our guides teach you about key features of Tune,
        such as distributed training or early stopping.


        +++
        .. button-ref:: tune-guides
            :color: primary
            :outline:
            :expand:

            Learn How To Use Tune

    .. grid-item-card::

        **Examples**
        ^^^

        In our examples you can find practical tutorials for using frameworks such as
        scikit-learn, Keras, TensorFlow, PyTorch, and mlflow, and state of the art search algorithm integrations.

        +++
        .. button-ref::  tune-examples-ref
            :color: primary
            :outline:
            :expand:

            Ray Tune Examples

    .. grid-item-card::

        **Ray Tune FAQ**
        ^^^

        Find answers to commonly asked questions in our detailed FAQ.

        +++
        .. button-ref:: tune-faq
            :color: primary
            :outline:
            :expand:

            Ray Tune FAQ

    .. grid-item-card::

        **Ray Tune API**
        ^^^

        Get more in-depth information about the Ray Tune API, including all about search spaces,
        algorithms and training configurations.

        +++
        .. button-ref:: tune-api-ref
            :color: primary
            :outline:
            :expand:

            Read the API Reference


Why choose Tune?
----------------

There are many other hyperparameter optimization libraries out there.
If you're new to Tune, you're probably wondering, "what makes Tune different?"

.. dropdown:: Cutting-Edge Optimization Algorithms
    :animate: fade-in-slide-down

    As a user, you're probably looking into hyperparameter optimization because you want to quickly increase your
    model performance.

    Tune enables you to leverage a variety of these cutting edge optimization algorithms, reducing the cost of tuning
    by `terminating bad runs early <tune-scheduler-hyperband>`_,
    :ref:`choosing better parameters to evaluate <tune-search-alg>`, or even
    :ref:`changing the hyperparameters during training <tune-scheduler-pbt>` to optimize schedules.

.. dropdown:: First-class Developer Productivity
    :animate: fade-in-slide-down

    A key problem with many hyperparameter optimization frameworks is the need to restructure
    your code to fit the framework.
    With Tune, you can optimize your model just by :ref:`adding a few code snippets <tune-tutorial>`.

    Also, Tune removes boilerplate from your code training workflow,
    supporting :ref:`multiple storage options for experiment results (NFS, cloud storage) <tune-storage-options>` and
    :ref:`logs results to tools <tune-logging>` such as MLflow and TensorBoard, while also being highly customizable.

.. dropdown:: Multi-GPU & Distributed Training Out Of The Box
    :animate: fade-in-slide-down

    Hyperparameter tuning is known to be highly time-consuming, so it is often necessary to parallelize this process.
    Most other tuning frameworks require you to implement your own multi-process framework or build your own
    distributed system to speed up hyperparameter tuning.

    However, Tune allows you to transparently :ref:`parallelize across multiple GPUs and multiple nodes <tune-parallelism>`.
    Tune even has seamless :ref:`fault tolerance and cloud support <tune-distributed-ref>`, allowing you to scale up
    your hyperparameter search by 100x while reducing costs by up to 10x by using cheap preemptible instances.

.. dropdown:: Coming From Another Hyperparameter Optimization Tool?
    :animate: fade-in-slide-down

    You might be already using an existing hyperparameter tuning tool such as HyperOpt or Bayesian Optimization.

    In this situation, Tune actually allows you to power up your existing workflow.
    Tune's :ref:`Search Algorithms <tune-search-alg>` integrate with a variety of popular hyperparameter tuning
    libraries (see :ref:`examples <tune-examples-ref>`) and allow you to seamlessly scale up your optimization
    process - without sacrificing performance.

Projects using Tune
-------------------

Here are some of the popular open source repositories and research projects that leverage Tune.
Feel free to submit a pull-request adding (or requesting a removal!) of a listed project.

- `Softlearning <https://github.com/rail-berkeley/softlearning>`_: Softlearning is a reinforcement learning framework for training maximum entropy policies in continuous domains. Includes the official implementation of the Soft Actor-Critic algorithm.
- `Flambe <https://github.com/asappresearch/flambe>`_: An ML framework to accelerate research and its path to production. See `flambe.ai <https://flambe.ai>`_.
- `Population Based Augmentation <https://github.com/arcelien/pba>`_: Population Based Augmentation (PBA) is a algorithm that quickly and efficiently learns data augmentation functions for neural network training. PBA matches state-of-the-art results on CIFAR with one thousand times less compute.
- `Fast AutoAugment by Kakao <https://github.com/kakaobrain/fast-autoaugment>`_: Fast AutoAugment (Accepted at NeurIPS 2019) learns augmentation policies using a more efficient search strategy based on density matching.
- `Allentune <https://github.com/allenai/allentune>`_: Hyperparameter Search for AllenNLP from AllenAI.
- `machinable <https://github.com/frthjf/machinable>`_: A modular configuration system for machine learning research. See `machinable.org <https://machinable.org>`_.
- `NeuroCard <https://github.com/neurocard/neurocard>`_: NeuroCard (Accepted at VLDB 2021) is a neural cardinality estimator for multi-table join queries. It uses state of the art deep density models to learn correlations across relational database tables.



Learn More About Ray Tune
-------------------------

Below you can find blog posts and talks about Ray Tune:

- [blog] `Tune: a Python library for fast hyperparameter tuning at any scale <https://towardsdatascience.com/fast-hyperparameter-tuning-at-scale-d428223b081c>`_
- [blog] `Cutting edge hyperparameter tuning with Ray Tune <https://medium.com/riselab/cutting-edge-hyperparameter-tuning-with-ray-tune-be6c0447afdf>`_
- [slides] `Talk given at RISECamp 2019 <https://docs.google.com/presentation/d/1v3IldXWrFNMK-vuONlSdEuM82fuGTrNUDuwtfx4axsQ/edit?usp=sharing>`_
- [video] `Talk given at RISECamp 2018 <https://www.youtube.com/watch?v=38Yd_dXW51Q>`_
- [video] `A Guide to Modern Hyperparameter Optimization (PyData LA 2019) <https://www.youtube.com/watch?v=10uz5U3Gy6E>`_ (`slides <https://speakerdeck.com/richardliaw/a-modern-guide-to-hyperparameter-optimization>`_)

Citing Tune
-----------

If Tune helps you in your academic research, you are encouraged to cite `our paper <https://arxiv.org/abs/1807.05118>`__.
Here is an example bibtex:

.. code-block:: tex

    @article{liaw2018tune,
        title={Tune: A Research Platform for Distributed Model Selection and Training},
        author={Liaw, Richard and Liang, Eric and Nishihara, Robert
                and Moritz, Philipp and Gonzalez, Joseph E and Stoica, Ion},
        journal={arXiv preprint arXiv:1807.05118},
        year={2018}
    }
