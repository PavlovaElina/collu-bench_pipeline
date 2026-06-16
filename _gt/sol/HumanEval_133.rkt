#lang racket
(define (sum_squares lst)
  (for/sum ([x (in-list lst)]) (let ([c (inexact->exact (ceiling x))]) (* c c))))
