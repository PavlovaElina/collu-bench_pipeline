#lang racket
(define (any_int a b c)
  (and (exact-integer? a) (exact-integer? b) (exact-integer? c)
       (or (= a (+ b c)) (= b (+ a c)) (= c (+ a b)))))
