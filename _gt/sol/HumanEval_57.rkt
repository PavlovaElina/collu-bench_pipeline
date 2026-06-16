#lang racket
(define (monotonic l)
  (or (apply <= l) (apply >= l)))
