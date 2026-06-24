# Understanding Time Series Data in TensorFlow for DHIS2 Users

> **Audience:** DHIS2 developers, implementers, and analysts who understand health information systems but are new to AI, TensorFlow, and Hugging Face.

- - -

# Introduction

If you've worked with DHIS2 for several years, you've probably become comfortable thinking about data in terms of:

*   Organisation Units (health facilities)
    
*   Data Elements
    
*   Category Option Combos
    
*   Periods (daily, weekly, monthly, yearly)
    
*   Datasets
    
*   Indicators
    
*   Analytics tables
    

Artificial Intelligence doesn't replace these concepts—it simply represents them differently.

The biggest hurdle for most DHIS2 developers is understanding **how machine learning sees your data**.

This guide bridges that gap.

- - -

# Part 1 — How DHIS2 Thinks About Data

Imagine a very simple dataset.

| Facility | Month | Malaria Cases |
| --- | --- | --- |
| Clinic A | Jan | 120 |
| Clinic A | Feb | 132 |
| Clinic A | Mar | 115 |
| Clinic A | Apr | 140 |

To us this is simply a table.

DHIS2 stores it as relational data.

Machine Learning doesn't.

Instead, it transforms this table into numbers arranged in mathematical structures.

- - -

# Part 2 — AI Doesn't Understand Tables

Humans see this:

| Month | Cases |
| --- | --- |
| Jan | 120 |
| Feb | 132 |
| Mar | 115 |
| Apr | 140 |

The AI only sees

```
[120,132,115,140]
```

Nothing else.

It doesn't know:

*   what malaria is
    
*   what months are
    
*   which clinic this came from
    

It only sees numbers.

Our job is to arrange those numbers so patterns become obvious.

- - -

# Part 3 — What is a Tensor?

People often hear the word **Tensor** and imagine something incredibly complicated.

It isn't.

A Tensor is simply a container for numbers.

Think of it like increasing dimensions.

- - -

## Scalar (0D Tensor)

One number.

```
120
```

- - -

## Vector (1D Tensor)

A list of numbers.

```
[120,132,115,140]
```

This might represent four months of malaria cases.

- - -

## Matrix (2D Tensor)

Rows and columns.

```
[
 [120],
 [132],
 [115],
 [140]
]
```

Or perhaps multiple measurements.

```
[
 [120,35],
 [132,40],
 [115,32],
 [140,45]
]
```

Where

```
[
Malaria Cases,
HIV Tests
]
```

Each row is one month.

- - -

## 3D Tensor

Now imagine multiple clinics.

```
Clinic A

[
 [120,35],
 [132,40],
 [115,32]
]

Clinic B

[
 [45,20],
 [51,18],
 [60,19]
]
```

Now the computer sees

```
Clinic
↓
Month
↓
Measurements
```

This is a 3-dimensional tensor.

- - -

# Part 4 — A DHIS2 Analogy

Suppose your DHIS2 database contains

```
Organisation Unit
Period
Data Element
Value
```

Machine learning simply reorganises it into

```
Organisation Unit
↓
Time
↓
Measurements
```

Nothing magical has happened.

The same data is simply packed differently.

- - -

# Part 5 — Why Rearrange It?

Because AI models look for patterns.

For example,

Instead of seeing

```
120
132
115
140
```

the model begins seeing

```
↑
↑
↓
↑
```

Eventually it learns things like

> Every rainy season malaria increases.

or

> Every December reporting decreases.

or

> Facility X behaves differently from Facility Y.

These are patterns.

- - -

# Part 6 — Time Series

Time Series simply means

> Numbers arranged in chronological order.

Example

```
Jan
Feb
Mar
Apr
```

becomes

```
120
132
115
140
```

Nothing more.

The order matters.

If we shuffled them

```
140
115
132
120
```

the meaning would disappear.

- - -

# Part 7 — Features

Suppose instead of only malaria cases we also have

| Month | Malaria | Rainfall | Temperature |
| --- | --- | --- | --- |
| Jan | 120 | 95  | 31  |
| Feb | 132 | 102 | 30  |
| Mar | 115 | 85  | 28  |

Now each month has three numbers.

```
[
 [120,95,31],
 [132,102,30],
 [115,85,28]
]
```

Each column is called a **Feature**.

Features are simply variables.

- - -

# Part 8 — Samples

Imagine 100 clinics.

Each clinic contributes one time series.

```
Clinic 1

Jan Feb Mar...

Clinic 2

Jan Feb Mar...

Clinic 3

Jan Feb Mar...
```

Each clinic becomes one **sample**.

Instead of training on one clinic,

we train on hundreds or thousands.

- - -

# Part 9 — Labels

Suppose we want to predict next month's malaria cases.

Input

```
Jan
Feb
Mar
Apr
```

Output

```
May
```

The historical months are the **Features**.

The value we want to predict is called the **Label**.

- - -

# Part 10 — Inference

People often confuse **training** with **inference**.

Training means

> Learning patterns.

Inference means

> Using those learned patterns.

Imagine training a new health worker.

Training takes months.

After training,

you ask

> "How many malaria cases do you think next month will have?"

They answer immediately.

That's inference.

Exactly the same idea.

- - -

# Part 11 — Where TensorFlow Fits

TensorFlow doesn't store your database.

It doesn't replace DHIS2.

TensorFlow is simply a mathematics engine.

Its job is

```
Numbers

↓

Tensors

↓

Mathematical Operations

↓

Predictions
```

- - -

# Part 12 — Where Hugging Face Fits

TensorFlow provides the engine.

Hugging Face provides pre-trained models.

Think of it this way.

TensorFlow is like buying an empty car chassis.

You still need to build the engine.

Hugging Face is like buying a complete vehicle that's already been trained to drive.

You simply use it.

- - -

# Part 13 — A Very Simple TensorFlow Example

Python

Run

```
import numpy as np

# Monthly malaria cases
cases = np.array([
    120,
    132,
    115,
    140,
    155,
    170
])

print(cases)
```

Output

```
[120 132 115 140 155 170]
```

That array is already a tensor (or can be converted into one).

- - -

# Part 14 — Converting to a Tensor

Python

Run

```
import tensorflow as tf

tensor = tf.constant(cases)

print(tensor)
```

Output

```
tf.Tensor([120 132 115 140 155 170],shape=(6,),dtype=int64)
```

Notice TensorFlow has added information:

*   shape
    
*   datatype
    

because computers need to know how the numbers are organised.

- - -

# Part 15 — A Slightly More Complex Example

Suppose each month has three measurements.

Python

Run

```
data = tf.constant([
    [120,95,31],
    [132,102,30],
    [115,85,28],
    [140,110,32]
])
```

TensorFlow reports

```
shape=(4,3)
```

Meaning

```
4 time periods×3 measurements
```

- - -

# Part 16 — The Typical AI Workflow

```
DHIS2 Database

↓

Extract data

↓

Clean missing values

↓

Convert to numbers

↓

Create tensors

↓

Train model

↓

Save model

↓

Run inference

↓

Predict future values
```

- - -

# Part 17 — How This Could Fit into a DHIS2 Application

Imagine a DHIS2 app that predicts stock shortages.

Every night it:

1.  Reads analytics from DHIS2
    
2.  Builds tensors
    
3.  Loads a pre-trained TensorFlow model
    
4.  Predicts next month's stock levels
    
5.  Displays warnings inside DHIS2
    

DHIS2 remains the system of record.

TensorFlow becomes the prediction engine.

- - -

# Part 18 — Key Terms

| DHIS2 Concept | AI Equivalent |
| --- | --- |
| Data Value | Number |
| Data Element | Feature |
| Dataset | Collection of Samples |
| Organisation Unit | Sample or Entity |
| Period | Time Step |
| Analytics Table | Training Data |
| Indicator | Derived Feature |
| Forecast | Prediction |
| Validation Rule | Model Evaluation |

- - -

# Final Thoughts

If you're coming from a DHIS2 background, you already understand **structured data** and **time series**. Machine learning doesn't require you to abandon that knowledge—it asks you to represent it differently.

The progression is surprisingly straightforward:

```
DHIS2 Table
        ↓
Numeric Arrays
        ↓
Tensors
        ↓
TensorFlow
        ↓
Machine Learning Model
        ↓
Predictions
```

Once this mental model "clicks," you'll find that TensorFlow and Hugging Face are far less mysterious. The real challenge shifts from understanding the tools to asking meaningful questions of your data, such as:

*   Can we predict malaria outbreaks?
    
*   Which facilities are likely to stock out of medicines?
    
*   Can we detect anomalous reporting automatically?
    
*   Can we estimate missing data?
    
*   Can we forecast disease burden six months ahead?
    

These are exactly the kinds of problems where combining DHIS2 data with TensorFlow and Hugging Face models can have a significant impact.

- - -

### Suggested Next Steps

Since your background is in DHIS2, health data, and software development, I'd recommend a learning path that builds naturally on what you already know:

1.  **Learn tensors** using only DHIS2-like data (time series, facilities, indicators).
    
2.  **Build simple TensorFlow forecasting models** using monthly health data.
    
3.  **Learn Hugging Face Pipelines** for ready-made models.
    
4.  **Run small open-source language models locally** on your Apple M4 using Docker or Python.
    
5.  **Connect TensorFlow and Hugging Face models to a DHIS2 application**, so predictions become part of your existing analytics workflow.
    

This approach avoids abstract AI examples and keeps everything grounded in the health information systems you're already familiar with.
