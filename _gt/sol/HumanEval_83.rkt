#lang racket
(define (starts_one_ends n)
  (if (= n 1) 1 (* 18 (expt 10 (- n 2)))))
