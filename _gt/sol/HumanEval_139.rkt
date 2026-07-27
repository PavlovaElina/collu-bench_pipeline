#lang racket
(define (special_factorial n)
  (define (fact k) (for/product ([i (in-range 1 (+ k 1))]) i))
  (for/product ([k (in-range 1 (+ n 1))]) (fact k)))
