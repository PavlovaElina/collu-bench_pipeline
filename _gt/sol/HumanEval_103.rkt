#lang racket
(define (rounded_avg n m)
  (if (> n m) -1
      (let ([avg (/ (for/sum ([i (in-range n (+ m 1))]) i) (- (+ m 1) n))])
        (string-append "0b" (number->string (inexact->exact (round avg)) 2)))))
